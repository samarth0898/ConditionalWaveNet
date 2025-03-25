import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn as nn
import argparse 


from model import WaveNet
from dataset import WavenetDataset

def train(model, dataloader):
    epochs = 10
    learning_rate, weight_decay = 0.001, 0
    optimizer = Adam(params=model.parameters(), lr = learning_rate, weight_decay=weight_decay)
    lr_scheduler = None
    crit = nn.CrossEntropyLoss()

    for epoch in range(epochs): 
        for i, (input, target) in enumerate(dataloader): 
            optimizer.zero_grad()
            pred = model(input)
            break 
        break
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
    # print('parameter count: ', model.parameter_count())
    train_path = r'/Users/samarththopaiah/Desktop/DeepLearning/LLMs/DeepGenerativeModels/GenerativeModels/Autoregressive/additional/pytorch-wavenet/train_samples/bach_chaconne/dataset.npz'
    test_path  = r''
    train_data = WavenetDataset(dataset_file = train_path,
                      item_length=model.receptive_field + model.output_length - 1,
                      target_length=model.output_length,
                      file_location='./train_samples/bach_chaconne',
                      test_stride=500)
    print('the dataset has ' + str(len(train_data)) + ' items')
    train_loader = DataLoader(dataset = train_data, batch_size = 32, shuffle = True, num_workers = 0, pin_memory = False)
    train(model, train_loader)
    
    

if __name__ == "__main__": 
    # parser = argparse.ArgumentParser()
    main()
