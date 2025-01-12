import torch
from tqdm import tqdm
from datetime import datetime
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from model.encoder import TSEncoder, ProjectionHead
from model.cl_loss import id_contrastive_loss
from utils import shuffle_feature_label
from utils import MyBatchSampler

class ISL:
    '''The ISL model.
    
    Args:
        input_dims (int): The input dimension. For a uni-variate time series, this should be set to 1.
        output_dims (int): The representation dimension.
        hidden_dims (int): The hidden dimension of the encoder.
        length (int): The length of the representation/series.
        depth (int): The number of hidden residual blocks in the encoder.
        device (str): The gpu used for training and inference.
        lr (float): The learning rate.
        batch_size (int): The batch size of samples.
        multi_gpu (bool): A flag to indicate whether using multiple gpus
        callback_func (Union[Callable, NoneType]): A callback function that would be called after each epoch.
    '''
    def __init__(
        self,
        input_dims=12,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device='cuda',
        lr=1e-4,
        batch_size=256,
        multi_gpu=True,
        callback_func=None
    ):
        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        
        self.multi_gpu = multi_gpu
        # gpu_idx_list = [0, 1]
        self.callback_func = callback_func
        
        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth)
      
        device = torch.device(device)
        if device == torch.device('cuda') and self.multi_gpu:
            # self.net_q = nn.DataParallel(self.net_q, device_ids=gpu_idx_list)
            self._net = nn.DataParallel(self._net)
        self._net.to(device)
        # stochastic weight averaging
        # https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
        # self.net = self._net
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)