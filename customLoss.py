import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math


# MSE
def MSE (y_pred, y_act, samples):
    # changes result from a numpy.float64 to pytorch data type
    delta = y_act - y_pred
    result = torch.mean(delta**2)/2
    return result
    # type = tensor(0.3026, dtype=torch.float64, grad_fn=<MeanBackward0>)

# MAPE
def MAPE (y_pred, y_act, samples):
    # changes result from a numpy.float64 to pytorch data type
    y_act = torch.tensor(y_act, dtype=torch.float64, requires_grad=True)
    eps = 1e-7
    delta = abs((y_act - y_pred)/(y_act + eps))
    result = 100*torch.mean(delta)/samples
    return result
    # type = tensor(0.3026, dtype=torch.float64, grad_fn=<MeanBackward0>)


def APE (y_pred, y_act):
    # changes result from a numpy.float64 to pytorch data type
    y_act = torch.tensor(y_act, dtype=torch.float64, requires_grad=True)
    delta = abs((y_act - y_pred)/(y_act))
    result = 100*delta
    return result

# MAE
def MAE (y_pred, y_act, samples):
    # changes result from a numpy.float64 to pytorch data type
    y_act = torch.tensor(y_act, dtype=torch.float64, requires_grad=True)
    delta = abs(y_act - y_pred)
    result = torch.mean(delta)/samples
    return result
    # type = tensor(0.3026, dtype=torch.float64, grad_fn=<MeanBackward0>)
