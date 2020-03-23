from overrides import overrides
import torch
from torch.nn.parameter import Parameter

from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention
from allennlp.nn import Activation


@MatrixAttention.register("plain_feedforword")
class Plain_Feedforward(MatrixAttention):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activation: Activation = None,
                 ) -> None:
        super().__init__()

        self._weight_matrix = Parameter(torch.Tensor(input_dim, output_dim))

        self._activation = activation or Activation.by_name('linear')()

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._weight_matrix)

    @overrides
    def forward(self, matrix: torch.Tensor) -> torch.Tensor:
        weight = self._weight_matrix
        return torch.matmul(self._activation(matrix) , weight)





@MatrixAttention.register("attention_feedforward")
class Attention_Feedforward(MatrixAttention):
    def __init__(self,
                 key_dim:int,
                 value_dim: int,
                 output_dim: int,

                 activation: Activation = None,
                 ) -> None:
        super().__init__()

        self._weight_modify = Parameter(torch.Tensor(value_dim, output_dim))
        self._weight_w = Parameter(torch.Tensor(key_dim+output_dim,output_dim))
        self._weight_v = Parameter(torch.Tensor(output_dim, 1))

        self._activation = activation or Activation.by_name('linear')()
        self.m_tanh = torch.nn.Tanh()

        self.reset_parameters()



    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._weight_modify)
        torch.nn.init.xavier_uniform_(self._weight_w)
        torch.nn.init.xavier_uniform_(self._weight_v)

    @overrides
    def forward(self, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        value = torch.matmul(self._activation(value) , self._weight_modify)
        #print('value size:', value.size())
        #print('key size:', key.size())
        value = value.unsqueeze(1) * torch.ones_like(key).to(key.device)
        tmp = torch.matmul(torch.cat((key,value) , -1), self._weight_w)
        tmp = self.m_tanh(tmp)
        attention = torch.matmul(tmp, self._weight_v).squeeze(-1)
        #print('attention size:', attention.size())
        attention = torch.nn.functional.softmax(attention, -1)
        #print('attention size after softmax:', attention.size())
        return torch.sum(attention.unsqueeze(-1) * value, -2, keepdim = True)
