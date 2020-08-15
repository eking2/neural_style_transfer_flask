import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models

class VGG(nn.Module):

    def __init__(self, layers_df):

        super().__init__()

        # feature maps to save
        self.select = layers_df['layer_idx'].values.tolist()

        # load vgg19 
        self.model = models.vgg19(pretrained=True).features

        # freeze params, not training model
        for param in self.model.parameters():
            param.requires_grad_(False)


    def forward(self, x):

        # get feature outputs from conv layers
        features = {}

        # ._modules is dict
        for layer_idx, layer in self.model._modules.items():
            x = layer(x)

            # save conv output for selected layers
            if layer_idx in self.select:
                features[layer_idx] = x

        return features


