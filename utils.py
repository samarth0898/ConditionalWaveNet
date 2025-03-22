
import numpy as np
import math
import torch 


def constant_pad_1d(input,
                    target_size,
                    value=0,
                    pad_start=False):
    """
         Assumes that padded dim is the 2, based on pytorch specification.
         Input: (N,C,Win)(N, C, W_{in})(N,C,Win​)
         Output: (N,C,Wout)(N, C, W_{out})(N,C,Wout​) where
     :param input:
     :param target_size:
     :param value:
     :param pad_start:
     :return:
     """
    num_pad = target_size - input.size(2)
    assert num_pad >= 0, 'target size has to be greater than input size'
    padding = (num_pad, 0) if pad_start else (0, num_pad)

    return torch.nn.ConstantPad1d(padding, value)(input)
                                                  
def dilate(x, dilation, init_dilation, pad_start = True):
    
    [n, c, l] = x.size() # N is the input dilation, C is the number of channels, and L is the input length
    dilation_factor = dilation / init_dilation
    
    # if there is no dilation change 
    if dilation_factor == 1:
        return x 

    # zero padding for reshaping
    new_l = int(np.ceil(l / dilation_factor) * dilation_factor)
    if new_l != l:
        l = new_l
        x = constant_pad_1d(x, new_l, pad_start=pad_start)

    l_old = int(round(l / dilation_factor))
    n_old = int(round(n * dilation_factor))
        
    l = math.ceil(l * init_dilation / dilation)
    n = math.ceil(n * dilation / init_dilation)

    # reshape according to dilation
    x = x.permute(1, 2, 0).contiguous()  # (n, c, l) -> (c, l, n)
    x = x.view(c, l, n)
    x = x.permute(2, 0, 1).contiguous()  # (c, l, n) -> (n, c, l)

    return x
    