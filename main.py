import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import argparse 
import os

from model import WaveNet
from dataset import WavenetDataset

# some ddp functionality ~ from Karpathy
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    dist.nit_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

def cleanup():
    dist.destroy_process_group()

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"


def generate_block(model, temperature = [0.5]): 
    model.eval() 
    samples, length = [], 80000 
    for temp in temperature: 
        samples.append(model.generate(model, num_samples = length, temperature=temp)) 
        samples = np.stack(samples, axis=0) 
        # tf_samples = tf.convert_to_tensor(samples, dtype=tf.float32) 
        return samples
def train(model, train_dataloader, clip = None):
    epochs = 10
    learning_rate, weight_decay = 0.001, 0
    optimizer = Adam(params=model.parameters(), lr = learning_rate, weight_decay=weight_decay)
    
    lr_scheduler = None
    verbose = True

    # some variables required 
    train_loss, val_loss, step = 0.0, 0.0, 0
    crit = nn.CrossEntropyLoss()
    
    in_dtype = torch.FloatTensor
    out_ltype = torch.LongTensor

    if master_process: 
        writer = SummaryWriter()

    for epoch in range(epochs): 
        loss_accum = 0.0
        for i, (x, target) in enumerate(train_dataloader): 
            # input : B, Quantize, Length
            # target: 

            x = Variable(x.type(in_dtype))
            target = Variable(target.view(-1).type(out_ltype))
            
            optimizer.zero_grad()
            output = model(x)
         
            loss = F.cross_entropy(output.squeeze(), target.squeeze())
            loss_accum += loss.detach()
            loss.backward()
            if ddp: 
                dist.all_reduce(loss_accum, op = dist.ReduceOp.AVG)
            
            # gradient clipping
            if clip: 
                nn.utils.clip_grad(model.parameters(), clip)
            optimizer.step()
            if master_process: 
                step += 1
                print(f'[{epoch}/{epochs}]|[{step}]|train loss {loss.item():04f}')
        
        if master_process and verbose: 
            loss_accum /= len(train_dataloader)
            print(f'[{epoch}/{epochs}] | train loss {loss_accum:04f}')
            writer.add_scalar('epoch_train_loss', loss_accum, epoch)
            train_loss = 0.0

        # checkpoint every 1000 steps 
        if (step % 1 == 0) and master_process: 
            ckpt = {'model': model.state_dict(), 'optim': optimizer.state_dict(), 'step': step}
            torch.save(ckpt, f'./checkpoints/{step}.pt')
    cleanup()

def main(): 
   
    model = WaveNet(num_layers = 10,
                     num_blocks=3,
                     dilation_channels=32,
                     residual_channels=32,
                     skip_channels=1024,
                     end_channels=512,
                     output_length=16,
                    #  dtype=torch.float32,
                     bias=True)
    
    
    
    training, generate = False, True
    
    if training:
        train_path = r'/Users/samarththopaiah/Desktop/DeepLearning/LLMs/DeepGenerativeModels/GenerativeModels/Autoregressive/additional/pytorch-wavenet/train_samples/bach_chaconne/dataset.npz'
        test_path  = r''

        train_data = WavenetDataset(dataset_file = train_path,
                        item_length=model.receptive_field + model.output_length - 1,
                        target_length=model.output_length,
                        file_location='./train_samples/bach_chaconne',
                        test_stride=500)
        if master_process:
            print('model: ', model)
            print('receptive field: ', model.receptive_field)
            print('the dataset has ' + str(len(train_data)) + ' items')
        if ddp: 
            sampler = DistributedSampler(dataset= train_data, rank= ddp_local_rank, num_replicas = ddp_world_size, shuffle=True)
            bs = 256 // 3
        else:
            sampler = None
            bs = 256
        train_loader = DataLoader(dataset = train_data, batch_size = bs, sampler=sampler)
        train(model, train_loader)

    if generate: 
        print('==> Load from pre-trained')
        ckpt = torch.load(r'./ckpt/9_23470.pt', weights_only=False)
        model.load_state_dict(ckpt['model'])

        generate_block(model = model)    

if __name__ == "__main__": 
    # parser = argparse.ArgumentParser()
    main()
