import torch.nn as nn
import torch


class update_a(nn.Module):
    def __init__(self, bands, P, w1_init, b1_init):
        super(update_a, self).__init__()
        self.w1 = nn.Conv1d(bands, P, 1,  bias=False)
        self.b1 = nn.Conv1d(P, P, 1,  bias=False)
        self._init_weights( w1_init, b1_init)

        pass
    def forward(self, v1, d1, v2, d2, v3, d3):

        h1 = v1 + d1
        h2 = v2 + d2
        h3 = v3 + d3

        h11 = self.w1(h1)
        h2 = self.b1(h2 + h3)

        a = h11 + h2
        return a

    def _init_weights(self, w1_init, b1_init):
        if w1_init.shape != self.w1.weight.shape or b1_init.shape != self.b1.weight.shape:
            raise ValueError("Shape mismatch: Custom matrix shape  does not match Conv1d weight shape .")
        self.w1.weight.data = w1_init
        self.b1.weight.data = b1_init
        pass




class update_v1(nn.Module):
    def __init__(self, bands, P, w2_init, b2_init, c2_init):
        super(update_v1, self).__init__()
        self.w2 = nn.Parameter(torch.Tensor([w2_init]), requires_grad=True)
        self.b2 = nn.Conv1d(P, bands, 1)
        self.c3 = nn.Parameter(torch.Tensor([c2_init]), requires_grad=True)
        self._init_weights(b2_init)

    def forward(self, y, a, d1):

        h2 = self.w2 * y
        h3 = self.b2(a)
        h4 = self.c3 * d1

        v1 = (h2 + h3 - h4)

        return v1

    def _init_weights(self, b2_init):

        self.b2.weight.data = b2_init


class CA(nn.Module):

    def __init__(self, in_ch):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 2, kernel_size=1, padding=0, bias=False),
            nn.LeakyReLU()
        )
        self.act = nn.Softmax(dim=-1)

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        y = y.transpose(-1, -2)

        m = self.max_pool(x)
        m = self.conv(m)
        m = m.transpose(-1, -2)
     
        y = self.act(y + m)

        return y


class update_v2(nn.Module):
    def __init__(self, P,  w3_w_init, w3_b_init, w3_c_init, kernel_class):
        super(update_v2, self).__init__()
        self.w3 = nn.Conv1d(P, P, 1, bias=False)


        if kernel_class == 'rbf':
            self.kernel_lays = GaussianKernelLayer(w3_c_init)
        else:
            raise ValueError('Not this kernel')

        self.b3 = nn.Conv1d(P, P, 1, bias=False)
        self.ca = CA(in_ch=P)

        self._init_weights(w3_w_init, w3_b_init)


    def forward(self, a, d2, y):
        h = a - d2
        h2 = self.w3(h)

        h3 = self.kernel_lays(y)

        pixel_num = h3.shape[2]
        ratio = self.ca(h3)

        h3 = torch.concat([h3[:, :, pixel_num // 2].unsqueeze(dim=1), h3.mean(dim=-1).unsqueeze(dim=1)], dim=1)

        b, _, P = h3.shape
        h3 = h3.reshape(b, 2, -1)

        h3 = torch.bmm(ratio, h3)

        h3 = h3.reshape(b, P).unsqueeze(dim=-1)

        h4 = self.b3(h3)
        v2 = h4 + h2

        return v2


    def _init_weights(self, w3_w_init, w3_b_init):

        self.w3.weight.data = w3_w_init
        self.b3.weight.data = w3_b_init



class GaussianKernelLayer(nn.Module):
    def __init__(self, w3_c_init, sigma=2.0):
        super(GaussianKernelLayer, self).__init__()
        self.w3_c = nn.Parameter(w3_c_init, requires_grad=True)
        self.sigma  = nn.Parameter(torch.Tensor([sigma]), requires_grad=True)

    def forward(self, x):
        batch = x.shape[0]
        k1 = x.shape[2]
        k2 = self.w3_c.shape[2]
        x = x.unsqueeze(dim=-1)
        w3_c2 = self.w3_c.unsqueeze(dim=-2)
        distances = torch.norm(x - w3_c2, dim=1, keepdim=True)
        gaussian_output = torch.exp(-0.5 * (distances / self.sigma).pow(2)).reshape(batch, k1, k2).permute(0, 2, 1)  # [256, P, 9]      

        return gaussian_output



class update_v3(nn.Module):
    def __init__(self,p,  init_w4):
        super(update_v3, self).__init__()
        self.w4 = nn.Parameter(torch.Tensor([init_w4]), requires_grad=True)
        self.act = nn.ReLU()

    def forward(self, a, d3):

        h1 = a - d3
        h1 = h1.abs()
        h2 = h1 - self.w4

        v3 = self.act(h2)

        return v3

class update_d1(nn.Module):
    def __init__(self, bands, P, w5_init):
        super(update_d1, self).__init__()
        self.w5 = nn.Conv1d(P, bands, 1, bias=False)
        self.b5 = nn.Conv1d(bands, bands, 1, bias=False)

        self._init_weights(w5_init)

    def forward(self, d1, a, v1):
        h1 = self.w5(a)
        h2 = h1 - v1
        h3 = self.b5(h2)

        d1 = d1 - h3

        return d1

    def _init_weights(self, w5_init):

        self.w5.weight.data = w5_init
        nn.init.constant_(self.b5.weight, 0.00)




class update_d23(nn.Module):
    def __init__(self, P):
        super(update_d23, self).__init__()

        self.w6 = nn.Conv1d(P, P, 1, bias=False)
        self._init_weights()
        pass

    def forward(self, d23, a, v23):
        h1 = a - v23
        h2 = self.w6(h1)
        d23 = d23 - h2

        return d23


    def _init_weights(self):

        nn.init.constant_(self.w6.weight, 0.00)




class bi_admm(nn.Module):
    def __init__(self,  bands, P,  w1_init, b1_init, w2_init, b2_init, c2_init, w3_w_init, w3_b_init, w3_c_init, init_w4, w5_init, kernel_class, stage=3):
        super(bi_admm, self).__init__()
        self.bands = bands
        self.P = P
        self.stage = stage
        self.module_list = nn.ModuleList([])

        for k in range(self.stage):

            self.module_list.append(update_a(bands, P, w1_init, b1_init))
            self.module_list.append(update_v1(bands, P, w2_init, b2_init, c2_init))
            self.module_list.append(update_v2(P, w3_w_init, w3_b_init, w3_c_init, kernel_class))
            self.module_list.append(update_v3(P, init_w4))
            self.module_list.append(update_d1(bands, P, w5_init))
            self.module_list.append(update_d23(P))
            self.module_list.append(update_d23(P))

        self.asc = nn.Softmax(dim=1)


    def forward(self, Y_patch,  v1, d1, v2, v3, d2, d3):
        pix_num = Y_patch.shape[2]
        y = Y_patch[:, :, pix_num // 2].unsqueeze(dim=-1)

        for k in range(self.stage):

            a = self.module_list[7 * k](v1, d1, v2, d2, v3, d3)

            v1 = self.module_list[7 * k + 1](y, a, d1)
            v2 = self.module_list[7 * k + 2](a, d2, Y_patch)
            v3 = self.module_list[7 * k + 3](a, d3)

            d1 = self.module_list[7 * k + 4](d1, a, v1)
            d2 = self.module_list[7 * k + 5](d2, a, v2)
            d3 = self.module_list[7 * k + 6](d3, a, v3)

        a = v3

        a = a / (a.sum(1,  keepdims=True) + 1e-10)

        return a



class Decoder(nn.Module):
    def __init__(self, P, in_ch, w3_c_init, kernel_class='rbf'):
        super(Decoder, self).__init__()
        self.l_de = nn.Linear(P, in_ch,  bias=False)

        if kernel_class == 'rbf':
            self.ul_de = GaussianKernelLayer(w3_c_init)

        else:
            raise ValueError('Not this kernel')
        self.ca = CA(in_ch=P)
    def forward(self, a, Y_patch):
        a = a.permute(0, 2, 1)

        y_rec = self.l_de(a).permute(0, 2, 1)
        a = a.permute(0, 2, 1)

        h = self.ul_de(Y_patch)
        prop = self.ca(h)
        pixel_num = h.shape[2]

        h = torch.concat(
            [h[:, :, pixel_num // 2].unsqueeze(dim=1), h.mean(dim=-1).unsqueeze(dim=1)], dim=1)

        b, _, P = h.shape

        h = torch.bmm(prop, h)
        h = h.reshape(b, P, 1)
        y_nl_rec = torch.bmm(h.permute(0, 2, 1), a)

        return y_rec, y_nl_rec


if __name__ == '__main__':

    a, v1, d1, v2, v3, d2, d3 = torch.randn((1, 3, 1)), torch.randn((1, 156, 1)), torch.randn((1, 156, 1)), torch.randn((1, 3, 1)), torch.randn((1, 3, 1)), torch.randn((1, 3, 1)), torch.randn((1, 3, 1))
    Y_patch = torch.randn((1, 156, 9))
    w_init, b_init, w3_w_init, w3_b_init, w5_init, b5_init = torch.randn((3, 156, 1)), torch.randn((3, 3, 1)), torch.randn((3, 3, 1)), torch.randn((3, 3, 1)), torch.randn((156, 3, 1)), torch.randn((156, 156, 1))
    w3_c_init = torch.randn((1, 156, 3)).float()

    net_encoder = bi_admm(156, 3,  w_init, b_init, 1, torch.randn((156, 3, 1)), 1, w3_w_init, w3_b_init,w3_c_init ,1, w5_init, kernel_class='rbf')
    net_decoder = Decoder(3, 156, w3_c_init)

    a = net_encoder(Y_patch,  v1, d1, v2, v3, d2, d3)
    y_rec, y_nl_rec = net_decoder(a, Y_patch)

    print(a.shape)
    print(y_rec.shape, y_nl_rec.shape)

