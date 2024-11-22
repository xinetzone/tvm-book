import numpy as np


class Normalize:
    @staticmethod
    def max_min(x: np.ndarray):
        min_ = x.min()
        max_ = x.max()
        if min_ == max_:
            return 0
        else:
            return (x - min_)/(max_ - min_)

def l2_loss(lvalue, rvalue):
    """L2 损失"""
    lvalue, rvalue = [np.array(kk).astype("float32") for kk in [lvalue, rvalue]]
    m = np.prod(lvalue.shape)
    assert  m == np.prod(rvalue.shape)
    lvalue = lvalue.flatten()
    rvalue = rvalue.flatten()
    lvalue = Normalize.max_min(lvalue)
    rvalue = Normalize.max_min(rvalue)
    # lvalue = lvalue / np.linalg.norm(lvalue)
    # rvalue = rvalue / np.linalg.norm(rvalue)
    loss = lvalue - rvalue
    out = np.dot(loss, loss.T)
    return out/m

def cosine_similarity(lvalue, rvalue):
    """余弦相似度"""
    lvalue, rvalue = [np.array(kk).astype("float32") for kk in [lvalue, rvalue]]
    assert np.prod(lvalue.shape) == np.prod(rvalue.shape)
    lvalue = lvalue.flatten()
    rvalue = rvalue.flatten()
    out = np.dot(lvalue, rvalue.T)
    if out == 0:
        return 0
    else:
        n1 = np.linalg.norm(lvalue)
        n2 = np.linalg.norm(rvalue)
        out /= n1*n2
        return np.clip(out, -1, 1)

def absolute_error(lvalue, rvalue):
    """绝对误差"""
    lvalue, rvalue = [np.array(kk).astype("float32") for kk in [lvalue, rvalue]]
    assert np.prod(lvalue.shape) == np.prod(rvalue.shape)
    lvalue = lvalue.flatten()
    rvalue = rvalue.flatten()
    error = np.abs(lvalue - rvalue)
    return error
