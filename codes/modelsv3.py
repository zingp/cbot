import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class TextTansformer(nn.modules):
    def __init__(self,):
        super(TextTansformer, self).__init__()
        self.encoder = nn.TransformerEncoder()
        self.decoder = nn.TransformerDecoder()
        pass