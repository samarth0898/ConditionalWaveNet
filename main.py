import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import argparse 


from model import WaveNet
from dataset import WavenetDataset

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

    for epoch in range(epochs): 
        for i, (x, target) in enumerate(train_dataloader): 
            # input : B, Quantize, Length
            # target: 

            x = Variable(x.type(in_dtype))
            target = Variable(target.view(-1).type(out_ltype))
            
            optimizer.zero_grad()
            output = model(x)
         
            loss = F.cross_entropy(output.squeeze(), target.squeeze())
            loss.backward()

            train_loss += loss.item()
            # gradient clipping
            if clip: 
                nn.utils.clip_grad(model.parameters(), clip)
            optimizer.step()
        
        if verbose: 
            train_loss /= len(train_dataloader)
            print(f'[{epoch}/{epochs}] | train loss {train_loss:04f}')
            train_loss = 0.0

        # checkpoint every 1000 steps 
        if step % 1000 == 0: 
            ckpt = {'model': model.state_dict(), 'optim': optimizer.state_dict(), 'step': step}
            torch.save(ckpt, f'./checkpoints/{step}.pt')

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
    
    
    print('model: ', model)
    print('receptive field: ', model.receptive_field)
   
    train_path = r'/Users/samarththopaiah/Desktop/DeepLearning/LLMs/DeepGenerativeModels/GenerativeModels/Autoregressive/additional/pytorch-wavenet/train_samples/bach_chaconne/dataset.npz'
    test_path  = r''

    train_data = WavenetDataset(dataset_file = train_path,
                      item_length=model.receptive_field + model.output_length - 1,
                      target_length=model.output_length,
                      file_location='./train_samples/bach_chaconne',
                      test_stride=500)
    print('the dataset has ' + str(len(train_data)) + ' items')
    train_loader = DataLoader(dataset = train_data, batch_size = 8, shuffle = True, num_workers = 0, pin_memory = False)
    train(model, train_loader)
    
    

if __name__ == "__main__": 
    # parser = argparse.ArgumentParser()
    main()
