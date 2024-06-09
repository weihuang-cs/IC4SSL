import os
import torch
import numpy as np
import torch.nn as nn

def one_hot(labels, num_classes: int, dtype: torch.dtype = torch.float):
    """
    For a tensor `labels` of dimensions B1[spatial_dims], return a tensor of dimensions `BN[spatial_dims]`
    for `num_classes` N number of classes.
    Example:
        For every value v = labels[b,1,h,w], the value in the result at [b,v,h,w] will be 1 and all others 0.
        Note that this will include the background label, thus a binary mask should be treated as having 2 classes.
    """
    assert labels.dim() > 0, "labels should have dim of 1 or more."

    # if 1D, add singelton dim at the end
    if labels.dim() == 1:
        labels = labels.view(-1, 1)

    sh = list(labels.shape)

    assert sh[1] == 1, "labels should have a channel with length equals to one."
    sh[1] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=1, index=labels.long(), value=1)

    return labels