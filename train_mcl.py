''' Train MCPS
'''
import os
import argparse
import numpy as np
from mcl import MCL
from data import load_data
from utils import seed_everything, get_device


parser = argparse.ArgumentParser(description='MCPS training')
parser.add_argument('--seed', type=int, default=42, help='random seed')
# for the data
parser.add_argument('--root', type=str, default='dataset', help='root directory of datasets')
parser.add_argument('--data', type=str, default='chapman', help='select pretraining dataset')
parser.add_argument('--length', type=int, default=600, help='length of each sample')
parser.add_argument('--overlap', type=float, default=0., help='overlap of each sample')
parser.add_argument('--depth', type=int, default=10, help='depth of the encoder')
parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension of the model')
parser.add_argument('--output_dim', type=int, default=320, help='output dimension of the model')
parser.add_argument('--momentum', type=float, default=0.999, help='momentum for the model')
parser.add_argument('--queue_size', type=int, default=16384, help='queue size for the model')
parser.add_argument('--masks', type=str, default='all_true', help='masks for the model')
# for the training
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--shuffle', type=str, default='random', help='way to shuffle the data')
parser.add_argument('--logdir', type=str, default='log', help='directory to save logs')
parser.add_argument('--checkpoint', type=int, default=1, help='save model after each checkpoint')
parser.add_argument('--multi_gpu', action='store_true', help='use multiple GPUs')
parser.add_argument('--verbose', type=int, default=1, help='print loss after each epoch')
# todo
# parser.add_argument('--resume', type=str, default='', help='resume training from a checkpoint')

args = parser.parse_args()

logdir = os.path.join(args.logdir, f'mcl_{args.data}_{args.seed}')
if not os.path.exists(logdir):
    os.makedirs(logdir)

def main(): 
    seed_everything(args.seed)
    print(f'=> Set seed to {args.seed}')
    
    X_train, _, _, y_train, _, _ = load_data(args.root, args.data, length=args.length, overlap=args.overlap, shuffle=True, task='cmsc')
    
    device = get_device()
    print(f'=> Running on {device}')
    
    model = MCL(
        input_dims=X_train.shape[-1],
        output_dims=args.output_dim,
        hidden_dims=args.hidden_dim,
        depth=args.depth,
        device=device,
        lr=args.lr,
        batch_size=args.batch_size,
        momentum=args.momentum,
        queue_size=args.queue_size,
        multi_gpu=args.multi_gpu,
        callback_func=pretrain_callback
    )
    
    print(f'=> Train MCL')
    loss_list = model.fit(
        X_train,
        y_train,
        shuffle_function=args.shuffle,
        masks=args.masks,
        epochs=args.epochs,
        verbose=args.verbose
        )
    # save training loss
    np.save(os.path.join(logdir, 'loss.npy'), loss_list)
    
    
def pretrain_callback(model, epoch, checkpoint=args.checkpoint):
    if (epoch+1) % checkpoint == 0:
        model.save(os.path.join(logdir, f'pretrain_{epoch+1}.pth'))
        

if __name__ == '__main__':
    main()


    
    
    
    
    
    



