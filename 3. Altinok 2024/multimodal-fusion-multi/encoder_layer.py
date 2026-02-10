import torch.nn as nn 
from cross_attention import CrossAttention
from layer_norm import LayerNorm 
from position_wise_feed_forward import PositionwiseFeedForward 

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden=768, drop_prob=0.5): 
        super(EncoderLayer, self).__init__() 
        
        self.cross_attention = CrossAttention(d_model) 
        self.norm1 = LayerNorm(dim=d_model) 
        self.dropout1 = nn.Dropout(p=drop_prob) 

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob) 
        self.norm2 = LayerNorm(dim=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob) 

    def forward(self, query, key): 
        _x = query 
        x = self.cross_attention(query, key)
        
        x = self.dropout1(x) 
        x = self.norm1(x + _x) 

        _x = x 
        x = self.ffn(x) 
        
        x = self.dropout2(x) 
        x = self.norm2(x + _x) 
        
        return x 