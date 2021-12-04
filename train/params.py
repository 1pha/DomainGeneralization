from models import models


# model params
use_gpu = True
dataset_mean = (0.5, 0.5, 0.5)
dataset_std = (0.5, 0.5, 0.5)

batch_size = 32
epochs = 1000
gamma = 10
theta = 0.01

# path params
data_root = "./data"
save_dir = "./experiment"
