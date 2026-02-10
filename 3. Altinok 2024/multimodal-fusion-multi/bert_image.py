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
        self.vit_model = torch.nn.Sequential(OrderedDict([*(list(vit_model.named_children())[:-1])]))
        for param in self.vit_model.parameters():
            param.requires_grad = True

        self.bert = RobertaModel.from_pretrained("FacebookAI/roberta-base")
        nhid = self.bert.config.hidden_size
        for param in self.bert.parameters():
            param.requires_grad = False
        n_layers = 12
        if nfinetune > 0:
            for param in self.bert.pooler.parameters():
                param.requires_grad = True
            for i in range(n_layers - 1, n_layers - 1 - nfinetune, -1):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = True

        self.cme_img = EncoderLayer(nhid)
        self.cme_trans = EncoderLayer(nhid)
        self.binary_class = nn.Linear(nhid * 2, 1)
        self.mmse_class = nn.Linear(nhid * 2, 4)
        self.sigmo = nn.Sigmoid()
        self.drop = nn.Dropout(0.5)

    def forward(self, spectos, input_ids, attention_mask):
        img_embeddings = self.vit_model(spectos)
        trans_embeddings = self.bert(input_ids, attention_mask=attention_mask)[0]
        crossed_img = self.cme_img(img_embeddings, trans_embeddings)
        crossed_trans = self.cme_trans(trans_embeddings, img_embeddings)
        crossed_img = torch.mean(crossed_img, dim=1) 
        crossed_trans = torch.mean(crossed_trans, dim=1)
        embeddings = torch.cat((crossed_img, crossed_trans), dim=1)
        mmse_logits = self.mmse_class(embeddings)
        class_label = self.binary_class(embeddings)
        class_label = self.sigmo(class_label)
        return class_label, mmse_logits