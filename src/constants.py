import numpy as np
import torch
import random
from torch import cuda

device = "cuda" if cuda.is_available() else "cpu"

# заморозка рандомайзеров
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True

# GLOBAL VARS
INF_IMG_SIZE = 224  # 720
IMG_SIZE = 224
BATCH_SIZE = 6
EPOCHS = 200
NUM_KEYPOINTS = 26 * 2  # 4 pairs each having x and y coordinates
CELL_NUM = 28
BEST_LOSS = 99.99

IMGPATH = "imagePath"
POINT = "points"
