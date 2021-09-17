import torch
from torch import nn

class Improved_Semhash(nn.Module):
    def __init__(self):
        super().__init__()
        self.threshold = 0.

    def saturating_sigmoid(self, x):
        value_1 = torch.ones_like(x)
        value_0 = torch.zeros_like(x)
        return torch.max(value_0, torch.min(value_1, 1.2*torch.sigmoid(x)-0.1))

    def forward(self, scores):
        B, clip_num, _ = scores.size()
        if self.training:
            gauss_nosie = torch.randn(B, clip_num, 1)
        else:
            gauss_nosie = torch.zeros(B, clip_num, 1)
        score_n = scores + gauss_nosie
        v1 = self.saturating_sigmoid(score_n)
        v2 = (score_n > self.threshold).float() - torch.sigmoid(score_n - self.threshold).detach() + torch.sigmoid(score_n - self.threshold)
        v2 += v1 - v1.detach()

        seed = torch.rand(1)
        return v1 if seed > 0.5 else v2

if __name__ == '__main__':
    a = torch.randn(2, 10, 1)
    semhash = Improved_Semhash()
    score = semhash(a)
    print(score)
