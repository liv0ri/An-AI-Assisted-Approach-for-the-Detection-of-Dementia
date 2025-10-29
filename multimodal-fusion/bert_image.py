import torch 
import torch.nn as nn 
from transformers import RobertaModel 
from collections import OrderedDict
import timm 
from encoder_layer import EncoderLayer

class BertImage(nn.Module): 
  def __init__(self, nfinetune=0): 
    super(BertImage, self).__init__() 

    vit_model = timm.create_model('vit_base_patch16_224', pretrained=True) 
    self.vit_model = torch.nn.Sequential(OrderedDict([*(list(vit_model.named_children())[:-1])])) # Remove final parameter.
    for param in self.vit_model.parameters(): # Iterates through all parameters of the ViT model.
      param.requires_grad = True # Sets all parameters of the ViT model to require gradients, meaning they will be updated during training.

    self.bert = RobertaModel.from_pretrained("FacebookAI/roberta-base")
    nhid = self.bert.config.hidden_size # Retrieves the hidden size from the RoBERTa model's configuration.

    for param in self.bert.parameters():
      param.requires_grad = False # Frozen and not updated during training to leverage pre-trained features without modifying them.
    n_layers = 12 
    if nfinetune > 0: # Checks if the 'nfinetune' argument is greater than 0.
      for param in self.bert.pooler.parameters(): # Iterates through parameters of the RoBERTa's pooler layer.
        param.requires_grad = True # Sets the pooler layer's parameters to require gradients, allowing them to be fine-tuned.
      for i in range(n_layers-1, n_layers-1-nfinetune, -1): # Loops backward from the last encoder layer for 'nfinetune' number of layers.
        for param in self.bert.encoder.layer[i].parameters(): # Iterates through parameters of the current encoder layer.
          param.requires_grad = True # Sets these encoder layer's parameters to require gradients, enabling fine-tuning for the specified top 'nfinetune' layers.

    self.cme_img = EncoderLayer(nhid) # Initializes an EncoderLayer for processing image embeddings, taking the RoBERTa hidden size as input dimension.
    self.cme_trans = EncoderLayer(nhid) # Initializes another EncoderLayer for processing text embeddings.
    self.binary_class = nn.Linear(nhid*2, 1) # Defines a linear layer for binary classification. 
    self.sigmo = nn.Sigmoid() # Defines a Sigmoid activation function, typically used in binary classification to squash the output of the linear layer into a probability between 0 and 1.
    self.drop = nn.Dropout(0.5) # Defines a Dropout layer with a dropout probability of 0.5.

  def forward(self, spectos, input_ids, attention_mask):
    img_embeddings = self.vit_model(spectos) 

    trans_embeddings = self.bert(input_ids, attention_mask=attention_mask)[0]

    crossed_img = self.cme_img(img_embeddings, trans_embeddings) # Passes image embeddings as query and text embeddings as key to the first EncoderLayer to allow image features to be influenced by text.
    crossed_trans = self.cme_trans(trans_embeddings, img_embeddings) # Passes text embeddings as query and image embeddings as key to the second EncoderLayer to allow text features to be influenced by images.
    
    crossed_img = torch.mean(crossed_img, dim=1) 
    crossed_trans = torch.mean(crossed_trans, dim=1)

    embeddings = torch.cat((crossed_img, crossed_trans), dim=1) # Concatenates the aggregated image and text embeddings along the feature dimension. 
    class_label = self.binary_class(embeddings) # Passes the concatenated embeddings through the linear classification layer to get raw logits.
    class_label = self.sigmo(class_label) # Applies the Sigmoid activation function to the logits to produce a probability between 0 and 1 for binary classification.
    return class_label 