import torch
from torchvision import transforms, models
import numpy as np
import pandas as pd
from PIL import Image
from io import StringIO
import json


def load_image(img_path, max_size=400, shape=None):

    '''convert image to tensor

    Args:
        img_path (str) : path to image
        max_size (int) : max dim size for image
        shape (tuple of ints) : (height, width) to resize to

    Returns:
        image (tensor) : image tensor 
    '''

    image = Image.open(img_path)

    # resize if image is larger than max_size
    if max(image.size) > max_size and shape is None:
        image = transforms.Resize(max_size)(image)

    if shape:
        image = transforms.Resize(shape)(image)

    # normalize with ImageNet stats
    transform = transforms.Compose([
        #transforms.Resize(size, interpolation=5),  # Lanczos
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.485, 0.456, 0.406),
                             std = (0.229, 0.224, 0.225))])

    # add batch dim
    image = transform(image).unsqueeze(0)

    return image


def tensor_to_image(tensor):

    '''convert tensor to image numpy array

    Args:
        tensor (tensor) : BCHW image tensor

    Returns:
        image (np array) : image as numpy array
    '''

    # remove batch dim, CHW to HWC
    image = tensor.to('cpu').clone().detach().numpy()
    image = image.squeeze().transpose(1, 2, 0)

    # denorm and clip out of range
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = image.clip(0, 1)

    return image


def vgg_layers():

    '''get VGG19 layer indices and names'''

    # load pretrained VGG19
    model = models.vgg19(pretrained=True).features

    # key = idx, value = name
    model_dict = {}
    outer = 1
    inner = 1

    model_lines = str(model).split('\n')
    for line in model_lines:

        # counter up inner for each conv
        if 'Conv2d' in line:
            layer_idx = line.split(')')[0].split('(')[1]
            layer_name = f'conv{outer}_{inner}'

            model_dict[layer_idx] = layer_name
            inner += 1

        # counter up outer and reset inner
        if 'MaxPool2d' in line:
            outer += 1
            inner = 1

    return model_dict


def layers_from_json(json_str):

    '''get layer indices to extract feature maps from user json input 

    Args:
        json_str (str) : input format is {layer name : weight}

    Returns:
        layer_df (dataframe) : dataframe with columns [layer_idx, layer_name, weight]
     '''

    # user input to df to merge on
    json_weights = json.loads(json_str)
    weights = pd.DataFrame.from_dict(json_weights, orient='index').reset_index()
    weights.columns = ['layer_name', 'weight']

    # vgg19 layers
    layers = pd.DataFrame.from_dict(vgg_layers(), orient='index').reset_index()
    layers.columns = ['layer_idx', 'layer_name']

    # merge and clean
    layers_df = weights.merge(layers, how='left')[['layer_idx', 'layer_name', 'weight']]

    return layers_df
