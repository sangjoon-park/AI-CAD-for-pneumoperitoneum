import torch
import torch.nn as nn
import numpy as np

class CNN_model(nn.Module):
    def __init__(self, base_model):
        super(CNN_model, self).__init__()
        self.base_model = base_model
        self.num_ftrs = self.base_model.fc.in_features

        self.head_0 = nn.Linear(self.num_ftrs, 1)
        self.head_1 = nn.Linear(self.num_ftrs, 1)
        self.head_2 = nn.Linear(self.num_ftrs, 1)

    def forward(self, x):
        x = nn.Sequential(*list(self.base_model.children())[:-1])(x)
        b, f_c, f_h, f_w = x.size()
        x = x.view(b, f_c)
        logits_0 = self.head_0(x)
        logits_1 = self.head_1(x)
        logits_2 = self.head_2(x)

        return [logits_0, logits_1, logits_2], x