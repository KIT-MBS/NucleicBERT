import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLossMultiClass(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True, ignore_index=None):
        super(FocalLossMultiClass, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        elif isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2).contiguous()  # N,C,H*W => N,H*W,C
            input = input.view(-1, input.size(2))  # N*H*W,C

        target = target.view(-1, 1)  # Flatten target to (N*H*W, 1)

        # Ignore specified indices in target
        if self.ignore_index is not None:
            valid_indices = target != self.ignore_index
            input = input[valid_indices.squeeze()]
            target = target[valid_indices.squeeze()]

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)  # Gather the log probabilities for the predicted class
        logpt = logpt.view(-1)

        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.type():
                self.alpha = self.alpha.type_as(input)
            at = self.alpha.gather(0, target.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
