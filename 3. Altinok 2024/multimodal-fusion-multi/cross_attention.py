import torch 
import torch.nn as nn 

class CrossAttention(nn.Module): 
  def __init__(self, input_dim):
    # Calculates attention scores
    super(CrossAttention, self).__init__() 
    self.input_dim = input_dim
    self.query = nn.Linear(input_dim, input_dim)
    self.key = nn.Linear(input_dim, input_dim)
    self.value = nn.Linear(input_dim, input_dim)
    
    self.softmax = nn.Softmax(dim=2) 
  def forward(self, Q, K): 
    queries = self.query(Q)
    keys = self.key(K)
    values = self.value(K) 

    scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim**0.5) 
                                                                           
    attention = self.softmax(scores) 
    weighted = torch.bmm(attention, values) 
    return weighted 