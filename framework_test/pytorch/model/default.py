import os
import sys
import torch.nn as nn

sys.path.append(os.path.abspath('.'))
from utils.utils import stringify

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        pass
    def forward(self, x):
        return

def test():
    m = model()
    print(m)

if __name__ == "__main__":
    test()