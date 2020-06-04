import cv2
import numpy as np


# Utils
# --------------------------
def save_training_checkpoint(torch_model, isBest, episode_count):
    """Save trained models
    """
    filename = 'model_' + str(episode_count) + '.ckpt'
    if isBest:
        filename = 'model_best.ckpt'
    torch.save(torch_model, filename)

def load_checkpoint(target_model, filename):
    """load trained models ==> target_model
    """
    target_model.load_state_dict(torch.load(filename))
    print('Models from {} loaded sucessfully'.format(filename))


def weight_init(m):
    """Initialize the network
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)