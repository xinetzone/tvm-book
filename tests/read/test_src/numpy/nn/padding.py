import numpy as np


def pad2d(X, ph, pw, val=0):
    """Pad X with the given value in 2-D

    ph, pw : height and width padding
    val : padding value, default 0
    """
    assert len(X.shape) >= 2
    nh, nw = X.shape[2], X.shape[3]
    pad_shape = *X.shape[:2], nh+ph*2, nw+pw*2, *X.shape[4:]
    padX = np.ones(pad_shape, X.dtype) * val
    padX[:, :, ph:-ph, pw:-pw] = X
    return padX
