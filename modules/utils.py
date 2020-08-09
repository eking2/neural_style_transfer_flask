import torch
from torchvision import transforms
import numpy as np
from PIL import Image

def load_image(img_path, max_size, shape):

    '''convert image to tensor

    Args:
        img_path (str) : path to image
        max_size (int) : max dim size for image

    Returns:
        image (tensor) : image tensor 
    '''

    image = Image.open(img_path)

    # resize if image is larger than max_size
    if max(image.size) < max_size:
        size = max_size
    else:
        size = max(image.size)

    # normalize with ImageNet stats
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=5),  # Lanczos
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

