import torch
import argparse 


from model import WaveNet



def train(): 
   
    model = WaveNet( num_layers = 10,
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


if __name__ == "__main__": 
    # parser = argparse.ArgumentParser()
    train()
