import torch
import numpy as np
import random
import os
SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')