import torch
import torch.nn as nn
from transformers import AutoModel


class TrustPredictor(nn.Module):
    def __init__(self, model_name, dropout = 0.4):
        super(TrustPredictor, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_dim = self.encoder.config.hidden_size
        self.dropout = nn.Dropout
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]
        output = self.encoder(input_ids, attention_mask).last_hidden_state[:,0,:]
        output = self.linear(output)

        return output

        
