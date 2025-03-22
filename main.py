import argparse 
from model import WaveNet

def train(): 
    model = WaveNet()
    
    print('model: ', model)
    print('receptive field: ', model.receptive_field)
    print('parameter count: ', model.parameter_count())


if __name__ == __main__: 
    parser = argparse.ArgumentParser()