import torch
import torch.nn.functional as F

def loss_sad_y(output, Y2):

    cos_sim = F.cosine_similarity(output, Y2, dim=1, eps=1e-10)
    cos_sim = cos_sim.clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.mean(torch.acos(cos_sim))


def loss_sad_e(e1, e2):

    return torch.mean(torch.acos(torch.sum(e1 * e2, dim=0) /
                          (torch.norm(e1, dim=0, p=2) * torch.norm(e2, dim=0, p=2))))

#Gaussian_kernel
def rbf_kernel(x1, x2, sigma=2.0):
    return torch.exp(- torch.norm(x1 - x2, dim=1) ** 2 / (2 * (sigma ** 2)))
