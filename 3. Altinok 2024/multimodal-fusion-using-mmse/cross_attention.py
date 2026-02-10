import torch 
import torch.nn as nn 

class CrossAttention(nn.Module): 
  def __init__(self, input_dim):
    super(CrossAttention, self).__init__() 
    self.input_dim = input_dim 
    self.query = nn.Linear(input_dim, input_dim) 
    self.key = nn.Linear(input_dim, input_dim) 
    self.value = nn.Linear(input_dim, input_dim) 
    self.softmax = nn.Softmax(dim=2) 
    
  def forward(self, Q, K): 
    queries = self.query(Q) # Applies the linear transformation to the query input Q to get query vectors.
    keys = self.key(K) # Applies the linear transformation to the key input K to get key vectors.
    values = self.value(K) # Applies the linear transformation to the key input K to get value vectors.

    scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim**0.5) 
    attention = self.softmax(scores) 
    weighted = torch.bmm(attention, values)
    return weighted 