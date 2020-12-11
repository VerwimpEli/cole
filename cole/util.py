import numpy as np
import random
import torch
import torch.utils.data
import torch.nn.functional as F
import torch.autograd.functional
from core import *


def build_model(path, data='mnist', device='cpu'):
    """
    Loads and returns model at a given path, and sends to device
    """
    model = MLP() if data == 'mnist' else get_resnet18()
    model.load_state_dict(torch.load(path))
    model.to(device)
    return model


def laplace_matrix(a):
    """
    Calculate the Laplace Matrix of a given matrix a.
    :param a: 2D numpy array
    :return: 2D numpy array with result
    """
    n, m = a.shape
    result = np.zeros((n-2, m-2))
    for i in range(1, n-1):
        for j in range(1, m-1):
            result[i-1, j-1] = a[i-1, j] + a[i+1, j] + a[i, j+1] + a[i, j-1] - 4*a[i, j]
    return result


def flatten_parameters(model):
    """
    Returns the model weights as a 1D tensor
    :return: 1D torch tensor with size N, with N the model parameters.
    """
    with torch.no_grad():
        flat = torch.cat([
            p.view(-1) for p in model.parameters()
        ])
    return flat


# TODO: find where this is used, bc don't see the use of it rn
def flatten_parameters_wg(model):
    return torch.cat([p.view(-1) for p in model.parameters()])


def assign_parameters(model, params):
    """
    Assigns a flat parameter tensor back to a model. Batch norm layers weights are skipped.
    :param model: model layout, params will be overwritten.
    :param params: 1D tensor
    :return model. This is the same model as the passed one, since no model copy is made.
    """
    state_dict = model.state_dict(keep_vars=False)
    with torch.no_grad():
        total_count = 0
        for param in state_dict.keys():
            if 'running_mean' in param or 'running_var' in param or 'num_batches_tracked' in param:
                continue
            param_count = state_dict[param].numel()
            param_size = state_dict[param].shape
            state_dict[param] = params[total_count:total_count+param_count].view(param_size).clone()
            total_count += param_count
        model.load_state_dict(state_dict)
    return model


class WeightPlane:
    def __init__(self, w1, w2, w3):
        """
        Constructs a 2D plane spanned by w2 - w1 and w3 - w1. Parameters should be 1D.
        """
        self.w1, self.w2, self.w3 = w1, w2, w3
        self.u = w2 - w1
        self.v = w3 - w1
        self.v -= torch.dot(self.u, self.v) * self.u / torch.dot(self.u, self.u)

    def xy_to_weights(self, x, y):
        """
        Transform 2D coordinates in the plane back to original coordinates.
        """
        return self.w1 + x * self.u + y * self.v

    def weights_to_xy(self):
        """
        :return: numpy 2D array with the coordinates of w1, w2 and w3 in the 2D plane
        """
        non_orth_v = self.w3 - self.w1
        w3_x = (torch.dot(self.u, non_orth_v) / torch.dot(self.u, self.u)).item()
        model_weights = [[0, 0], [1, 0], [w3_x, 1]]
        return np.array(model_weights)


