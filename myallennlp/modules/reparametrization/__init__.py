from myallennlp.modules.reparametrization.gumbel_softmax import *


import torch

def data_dropout(data:torch.Tensor,frequency)->torch.Tensor:
    if frequency == 0: return data
    unk_mask = torch.bernoulli(torch.ones(data.size(),device=data.device)*frequency).type_as(data)
    data = data*(1-unk_mask)           #+(unk_mask*torch.ones(data.size()).type_as(data)*0).long()
    return data