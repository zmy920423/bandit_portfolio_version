import torch as torch


BATCH_SIZE = 100000
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

EVAL_DISPLAY_BATCH = 100

IDX_M = 100
