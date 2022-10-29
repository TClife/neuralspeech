import os 
import torch
import time 
import argparse 
from hparams import hparams as hps
from model.model import Tacotron2, Tacotron2Loss 
import numpy as np 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
def train(args):
    rank = args.rank
    n_gpus = args.num_gpus
    if 'WORLD_SIZE' in os.environ: #parallel training
        os.environ['OMP_NUM_THREADS'] = str(hps.n_workers)
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ('LOCAL_RANK'))
        n_gpus = int(os.environ['WORLD_SIZE'])
        torch.distributed.init_process_group(
            backend = 'nccl', rank = local_rank, world_size = n_gpus
        )
    torch.cuda.set_devices(local_rank) 
    device = torch.device('cuda:{:d}'.format(local_rank))
    
    #build model 
    model - Tacotron2()
     
 
 
 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d', '--data', type=str, default="/home/hschung/ecg/DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/LJSpeech-1.1/",
                        help = 'data directory')
    parser.add_argument('-l', '--log', type=str, default='log', 
                        help = 'tensorboad logs')
    parser.add_argument('--c', 'ckpt_dir', type=str, default = 'ckpt', 
                        help = "directory to save checkpoints") 
    parser.add_argument('--p', 'ckpt_path', type=str, default='',
                        help = "load checkpoints")
    parser.add_argument('--r', 'rank', type=int, default=0)
    parser.add_argument('--n', 'num_gpus', type=int, default=1)
    
    args = parser.parse_args()
    
    #train
    train(args) 
     
    
    
     
