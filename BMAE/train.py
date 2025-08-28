import torch.optim as opt
from loadhsi import *
import torch.utils.data as data
from lossFunction import *
from initialization import initial
from metric import *
from net import *
from tqdm import tqdm
import os

def extract_patches(input_tensor: torch.Tensor, kernel=3, stride=1, pad_num=0):
    if pad_num != 0:
        input_tensor = torch.nn.ReflectionPad2d(pad_num)(input_tensor)
    all_patches = input_tensor.unfold(2, kernel, stride).unfold(3, kernel, stride)
    N, C, H, W, h, w = all_patches.shape
    all_patches = all_patches.permute(0, 2, 3, 1, 4, 5)
    all_patches = torch.reshape(all_patches, shape=(N * H * W, C, h, w))
    return all_patches

class PatchDataset(data.Dataset):

    def __init__(self, hsi: torch.Tensor,  kernel, stride):
        super(PatchDataset, self).__init__()
        self.hsi = extract_patches(hsi, kernel , stride , pad_num=(kernel - 1) // 2)
        self.num = self.hsi.shape[0]

    def __getitem__(self, item):
        hsi = self.hsi[item, :, :, :]

        return hsi, item

    def __len__(self):
        return self.num

def setup_seed(seed):
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train(dataloader, dataloader2,  E_init, alpha, mu, lamb, kernel_class, Bands, P, E_true, A_true, device, case):
    if case == 'samson':
        epochs = 300
        lr_encoder = 1e-4
        lr_decoder = 1e-4
        weight_decay = 1e-3
        gama = 0.96
        E_init = E_init / np.max(E_init, axis=0)
    elif case == 'Apex':
        epochs = 300
        lr_encoder = 1e-4
        lr_decoder = 1e-4
        weight_decay = 1e-5
        gama = 0.88
        E_init = E_init / np.max(E_init, axis=0)
    else:
        raise ValueError('None dataset')

    w1_init, b1_init, w2_init, c2_init, b2_init, w3_init, b3_init, c3_init, w4_init, w5_init = initial(E_init, alpha,
                                                                                            mu, lamb, kernel_class)

    w1_init = torch.as_tensor(w1_init).float().unsqueeze(dim=-1).to(device)
    b1_init = torch.as_tensor(b1_init).float().unsqueeze(dim=-1).to(device)
    b2_init = torch.as_tensor(b2_init).float().unsqueeze(dim=-1).to(device)
    w3_init = torch.as_tensor(w3_init).float().unsqueeze(dim=-1).to(device)
    b3_init = torch.as_tensor(b3_init).float().unsqueeze(dim=-1).to(device)
    c3_init = torch.as_tensor(c3_init).float().unsqueeze(dim=0).to(device)
    w5_init = torch.as_tensor(w5_init).float().unsqueeze(dim=-1).to(device)
    admm = bi_admm(Bands, P, w1_init, b1_init, w2_init, b2_init, c2_init, w3_init, b3_init, c3_init, w4_init, w5_init,
                   kernel_class, stage=6).to(device)
    decoder = Decoder(P, Bands, c3_init).to(device)

    decoder.l_de.weight.data = torch.as_tensor(E_init).float().to(device)
    optim1 = opt.Adam(admm.parameters(), lr=lr_encoder)
    sch1 = opt.lr_scheduler.StepLR(optim1, step_size=10, gamma=gama)
    optim2 = opt.Adam(decoder.parameters(), lr=lr_decoder, weight_decay=weight_decay)
    sch2 = opt.lr_scheduler.StepLR(optim2, step_size=10, gamma=gama)
    for iter in tqdm(range(epochs)):
        loss_sum = 0
        for i, y in enumerate(dataloader):
            admm.train()
            decoder.train()
            batch, Bands, row, col = y[0].shape
            y = y[0].to(device).reshape(batch, Bands, -1)
            patch_size = y.shape[2]

            v1, d1, v2, v3, d2, d3 = torch.zeros_like(y[:, :, patch_size // 2].unsqueeze(dim=-1)).to(device), \
                                     torch.zeros_like(y[:, :, patch_size // 2].unsqueeze(dim=-1)).to(device), \
                                     torch.zeros((batch, P, 1)).to(device), torch.zeros((batch, P, 1)).to(device), \
                                     torch.zeros((batch, P, 1)).to(device), torch.zeros((batch, P, 1)).to(device)
            a = admm(y, v1, d1, v2, v3, d2, d3)

            y_rec, y_nl_rec = decoder(a, y)

            E = decoder.l_de.weight.data
            E_nl = decoder.ul_de.w3_c.squeeze(dim=0).data

            y_c = y[:, :, patch_size // 2].unsqueeze(dim=-1)
            loss1 = loss_sad_y(y_c, y_rec)
            loss2 = torch.abs(y_nl_rec - rbf_kernel(y_c, y_c)).mean()
            loss3 = loss_sad_e(E, E_nl)

            loss =  loss1 + loss2  + loss3 * 0.02

            loss_sum += loss

            decoder.l_de.weight.data.clamp_(1e-10, 1.0)
            decoder.ul_de.w3_c.data.clamp_(1e-10, 1.0)

            optim1.zero_grad()
            optim2.zero_grad()
            loss.backward()

            optim2.step()
            optim1.step()


        sch2.step()
        sch1.step()

        if iter != 0:
            print('loss:{}'.format(loss_sum.item()))

    admm.eval()
    decoder.eval()
    with torch.no_grad():
        A_test = torch.empty([0, P]).to(device)
        y_est = torch.empty([0, Bands]).to(device)
        y_nl_est = torch.empty([0, 1]).to(device)
        for i, y in enumerate(dataloader2):
            batch, Bands, row, col = y[0].shape
            y = y[0].to(device).reshape(batch, Bands, -1)

            v1, d1, v2, v3, d2, d3 = torch.zeros_like(y).to(device), torch.zeros_like(y).to(
                device), torch.zeros((batch, P, 1)).to(device), torch.zeros((batch, P, 1)).to(
                device), torch.zeros((batch, P, 1)).to(device), torch.zeros((batch, P, 1)).to(device)

            A_t = admm(y, v1, d1, v2, v3, d2, d3)
            y_rec, y_nl_rec = decoder(A_t, y)
            y_c, A_c = y_rec[:, :, patch_size // 2], A_t[:, :, patch_size // 2]
            y_nl_c = y_nl_rec[:, : , patch_size // 2]
            y_est = torch.concat([y_est, y_c], dim=0)
            A_test = torch.concat([A_test, A_c], dim=0)
            y_nl_est = torch.concat([y_nl_est, y_nl_c])

        row, col = A_true.shape[1], A_true.shape[2]
        A_test = A_test.T.detach().cpu().numpy().reshape(-1, row, col)
        E_test = decoder.l_de.weight.data.detach().cpu().numpy()

        PLOT = False
        RMSE(A_true, A_test, P)
        SAD(E_test, E_true, P)
        if PLOT is True:
            draw_A(A_true, A_test, P)
            draw_E(E_true, E_test, P)

        weights_dir = './weights'
        case_dir = case + '_weights'
        weights_path = os.path.join(weights_dir, case_dir)
        os.makedirs(weights_path, exist_ok=True)
        admm_weights_path = os.path.join(weights_path, '_encoder_weights.pth')
        decoder_weights_path = os.path.join(weights_path,  '_decoder_weights.pth')
        torch.save(admm.state_dict(), admm_weights_path)
        torch.save(decoder.state_dict(), decoder_weights_path)
        sio.savemat(case + '_res.mat', {'E': E_test, 'A': A_test})

def main():
    setup_seed(0)
    case = 'samson'
    Y, A_true, E_true, P, N, Bands = loadhsi(case)

    alpha, mu, lamb, kernel_class = 0.5, 1e-3, 1e-4, 'rbf'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    Y3 = torch.as_tensor(Y.transpose(2, 0, 1)).unsqueeze(dim=0)
    dataset = PatchDataset(Y3, kernel=3, stride=1)


    dataloader = data.DataLoader(dataset, batch_size=512, shuffle=True, drop_last=True)
    dataloader2 = data.DataLoader(dataset, batch_size=512, shuffle=False, drop_last=False)

    E_init = sio.loadmat('vca_init/' + case + '_init.mat')['E']

    train(dataloader, dataloader2,  E_init, alpha, mu, lamb, kernel_class, Bands, P, E_true, A_true, device, case)


if __name__ == '__main__':
    main()

