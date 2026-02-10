import torch.nn as nn 

class PositionwiseFeedForward(nn.Module): 
    def __init__(self, d_model, hidden, drop_prob=0.5): 
        super(PositionwiseFeedForward, self).__init__()  
        
        self.linear1 = nn.Linear(d_model, hidden) # The first linear layer. It projects the input from `d_model` dimensions to `hidden` dimensions.
        self.linear2 = nn.Linear(hidden, d_model) # The second linear layer. It projects the output from `hidden` dimensions back to `d_model` dimensions.
        
        self.relu = nn.ReLU() # The ReLU activation function, applied after the first linear layer. It introduces non-linearity.
        self.dropout = nn.Dropout(p=drop_prob) # A Dropout layer to prevent overfitting by randomly setting a fraction of input units to zero during training.

    def forward(self, x): 
        x = self.linear1(x) # Passes the input 'x' through the first linear layer.
        x = self.relu(x) # Applies the ReLU activation function element-wise to the output of the first linear layer.
        x = self.dropout(x) # Applies dropout to the activated output.
        x = self.linear2(x) # Passes the result through the second linear layer, transforming it back to `d_model` dimensions.
        
        return x 