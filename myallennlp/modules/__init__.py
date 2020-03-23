from overrides import overrides
import torch
from torch.nn.parameter import Parameter

from allennlp.modules.attention.attention import Attention
from allennlp.nn import Activation

from myallennlp.modules.feedforward_as_seq2seq import MyFeedForward
from myallennlp.modules.bilinear_matrix_attetion_low_rank import BilinearMatrixAttention_Lowrank, BilinearMatrix
from myallennlp.modules.masked_softmax import MaskedOperation
from myallennlp.modules.massage_passing import Plain_Feedforward, Attention_Feedforward