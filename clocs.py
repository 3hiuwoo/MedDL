import torch
from tqdm import tqdm
from datetime import datetime
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from model.encoder import TSEncoder, CLEncoder
from model.cl_loss import id_contrastive_loss
from utils import shuffle_feature_label
from utils import MyBatchSampler


class CLOCS:
    '''The CLOCS model.
    
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
        
        self._net = CLEncoder(TSEncoder, input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth)
      
        device = torch.device(device)
        if device == torch.device('cuda') and self.multi_gpu:
            # self.net_q = nn.DataParallel(self.net_q, device_ids=gpu_idx_list)
            self._net = nn.DataParallel(self._net)
        self._net.to(device)
        # stochastic weight averaging
        # https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
        # self.net = self._net
        self.net = torch.optim.swa_utils.AveragedModel(self._net.encoder)
        self.net.update_parameters(self._net.encoder)
        
        
    def fit(self, X, y, shuffle_function='trial', masks=None, epochs=None, verbose=True):
        ''' Training the CLOCS model.
        
        Args:
            X (numpy.ndarray): The training data. It should have a shape of (n_samples, sample_timestamps, features).
            y (numpy.ndarray): The training labels. It should have a shape of (n_samples, 3). The three columns are the label, patient id, and trial id.
            shuffle_func (str): specify the shuffle function.
            masks (list): A list of masking functions applied (str).
            epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            epoch_loss_list: a list containing the training losses on each epoch.
        '''
        assert X.ndim == 4
        assert y.shape[1] == 3
        assert X.shape[-1] == 1
        print(f'=> Number of views: {X.shape[1]}')
        
        # Shuffle the training set for contrastive learning pretraining.
        X, y = shuffle_feature_label(X, y, shuffle_function=shuffle_function, batch_size=self.batch_size)

        # we need patient id for patient-level contrasting and trial id for trial-level contrasting
        train_dataset = TensorDataset(
            torch.from_numpy(X).to(torch.float),
            torch.from_numpy(y).to(torch.long)
            )
        
        if shuffle_function == 'random':
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        else:
            # Important!!! A customized batch_sampler to shuffle samples before each epoch. Check details in utils.py.
            my_sampler = MyBatchSampler(range(len(train_dataset)), batch_size=self.batch_size, drop_last=True)
            train_loader = DataLoader(train_dataset, batch_sampler=my_sampler)
        
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        
        epoch_loss_list = []
        start_time = datetime.now()           
        for epoch in range(epochs):
            cum_loss = 0
            for x, y in tqdm(train_loader, desc=f'=> Epoch {epoch+1}', leave=False):
                # count by iterations
                x = x.to(self.device)
                pid = y[:, 1]  # patient id

                optimizer.zero_grad()
                
                views = self._net(x, masks=masks)
                loss = id_contrastive_loss(views, pid)
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net.encoder)

                cum_loss += loss.item()
           
            cum_loss /= len(train_loader)
            epoch_loss_list.append(cum_loss)
            
            if verbose:
                print(f"=> Epoch {epoch+1}: loss: {cum_loss}")
                
            if self.callback_func is not None:
                self.callback_func(self, epoch)
                
        end_time = datetime.now()
        print(f'=> Training finished in {end_time - start_time}')
            
        return epoch_loss_list
    
    
    def save(self, fn):
        '''Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), fn)
    
    
    def load(self, fn):
        '''Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        # state_dict = torch.load(fn, map_location=self.device)
        state_dict = torch.load(fn)
        self.net.load_state_dict(state_dict)