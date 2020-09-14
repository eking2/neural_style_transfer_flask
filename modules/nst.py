import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import decimal
from .model import VGG
from .utils import load_image, tensor_to_image, layers_from_json


def gram_matrix(tensor):

    '''calculate feature correlations with gram matrix

    Args:
        tensor (tensor) : conv feature tensor

    Returns:
        gram (tensor) : gram matrix
    '''

    # get size of each dim and flatten HW
    b, c, h, w = tensor.shape
    tensor = tensor.view(c, h*w)

    # gram matrix is matrix mult inverse
    gram =  torch.mm(tensor, tensor.t())

    return gram


def resize_images(content_path, style_path, size):

    '''resize content and style images to identical shapes (content shape)

    Args:
        content_path (str) : path to content image
        style_path (str) : path to style image
        size (int) : size to rescale for smaller edge of image

    Returns:
        content (tensor) : content image tensor
        style (tensor) : style image tensor
    '''

    content = load_image(content_path, size)
    style = load_image(style_path, shape=(content.shape[2], content.shape[3]))

    return content, style


def style_transfer(content, style, model, steps, alpha, beta, layers_df):
    
    '''run neural style transfer model

    Args:
        content (tensor) : content image tensor
        style (tensor) : style image tensor
        model (model) : pytorch VGG model 
        steps (int) : number of steps to run
        alpha (float) : content reconstruction weight
        beta (float) : style reconstruction weight
        layers_df (dataframe) : layer weights
    '''

    # initialize generated image by cloning content and optimizer
    target = content.clone().requires_grad_(True)
    optimizer = optim.Adam([target], lr=3e-3)

    # save 5 intermediate images + 2 end points, equally spaced
    steps_to_save = torch.linspace(0, steps, 7).round().long()

    for step in range(steps+1):
        print(step)

        # get features
        target_features = model(target)
        content_features = model(content)
        style_features = model(style)

        # init losses
        content_loss = 0
        style_loss = 0

        # each layer style loss
        for idx, row in layers_df.iterrows():
            layer_idx = row['layer_idx']
            weight = row['weight']

            # content mse loss, paper originally uses mse of conv4_2 output
            content_loss += F.mse_loss(target_features[layer_idx], content_features[layer_idx])

            # style loss
            target_tensor = target_features[layer_idx]
            target_gram = gram_matrix(target_tensor)

            style_tensor = style_features[layer_idx]
            style_gram = gram_matrix(style_tensor)

            layer_loss_weighted = weight * F.mse_loss(target_gram, style_gram)

            # normalize loss
            b, c, h, w = target_tensor.shape
            style_loss += layer_loss_weighted / (c * h * w)

        # total loss
        total_loss = alpha * content_loss + beta * style_loss

        # update target image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # denorm to undo transforms
        if step in steps_to_save:
            save_path = f'static/output_{step}.png'
            target_img = tensor_to_image(target)
            plt.imshow(target_img)
            plt.axis('off')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()


def test():


    steps = 2_000
    size = 200
    alpha = 1
    beta = 1e6

    weights = '''{"conv1_1" : 1,
    "conv2_1" : 0.75,
    "conv3_1" : 0.2,
    "conv4_1" : 0.2,
    "conv5_1" : 0.2}'''

    layers_df = layers_from_json(weights)

    content_path = '../test_images/YellowLabradorLooking_new.jpg'
    style_path = '../test_images/Vassily_Kandinsky,_1913_-_Composition_7.jpg'
    content, style = resize_images(content_path, style_path, size)
    print('content:', content.shape)
    print('style:', style.shape)

    #plt.imshow(content.cpu().numpy().squeeze().transpose(1, 2, 0))
    #plt.savefig('test_content.png')
    #plt.close()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    content = content.to(device)
    style = style.to(device)
    model = VGG(layers_df).to(device)

    style_transfer(content, style, model, steps, alpha, beta, layers_df)

#test()




