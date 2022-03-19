import numpy as np
import torch

def MinMaxScale(X, y):
    MIN0 = np.min(y[:, 0])
    MIN1 = np.min(y[:, 1])
    MAX0 = np.max(y[:, 0])
    MAX1 = np.max(y[:, 1])
    
    X[:, 0] = (X[:, 0] - MIN0)/(MAX0 - MIN0)
    y[:, 0] = (y[:, 0] - MIN0)/(MAX0 - MIN0)
    X[:, 1] = (X[:, 1] - MIN1)/(MAX1 - MIN1)
    y[:, 1] = (y[:, 1] - MIN1)/(MAX1 - MIN1)
    
    return MIN0, MAX0, MIN1, MAX1, X, y

# Tensor 형태로 변환
def make_Tensor(array):
    return torch.from_numpy(array).float()