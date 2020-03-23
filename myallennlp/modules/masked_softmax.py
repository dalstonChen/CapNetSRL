
import torch
import torch.nn as nn


class MaskedOperation(nn.Module):

    @staticmethod
    def log_softmax(x, dim, mask=None):
        """
        Performs masked softmax, as simply masking post-softmax can be
        inaccurate
        :param x: [batch_size, num_items]
        :param mask: [batch_size, num_items]
        :return:
        """
        if mask is not None:
            mask = mask.float()
        if mask is not None:
            x_masked = x * mask + (1 - 1 / mask)
        else:
            x_masked = x
        res = torch.nn.functional.log_softmax(x_masked, dim)
        if mask is not None:
            res = res * mask.float()
        return res