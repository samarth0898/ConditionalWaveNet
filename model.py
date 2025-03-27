"""
Implementation of WaveNet, an autoregressive generative model 

1. Causal Conv (1x1)
2. Residual Block x num_blocks
    Dilated Conv
    Residual connections 
        tanh * sigmoid 
        Casual Conv (1x1) --> Skip connection
        + Residual connection 
3. Skip connection 

"""
import numpy as np

import torch.nn as nn 
import torch.nn.functional as F
import torch, torchaudio as ta
from torch.autograd import Variable

from utils import dilate
from dataset import mu_law_expansion

dilation_channels=32
residual_channels=32
skip_channels=256
end_channels=256
classes = 256 # U law companding, 256 levels with the softmax output 
kernel_size = 2 

# model = WaveNetModel(layers=10,
#                      blocks=3,
#                      dilation_channels=32,
#                      residual_channels=32,
#                      skip_channels=1024,
#                      end_channels=512,
#                      output_length=16,
#                      dtype=dtype,
#                      bias=True)

class WaveNet(nn.Module): 
    
    def __init__(self, residual_channels = 32, dilation_channels = 32, classes = 256,
                 skip_channels = 256, num_blocks = 3, num_layers = 10, output_length = 32, end_channels = 256, bias = True):
        super(WaveNet, self).__init__() 

        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.kernel_size = kernel_size

        input_channels = classes # why are the input channels 256? U law companding?

        self.causal_conv = nn.Conv1d(in_channels = input_channels, out_channels = residual_channels, kernel_size=1, bias=bias)

        
        self.dilation = []
        init_dilation, receptive_field = 1, 1

        # parametrized residual connection 
        self.filter = nn.ModuleList()
        self.gate   = nn.ModuleList()

        self.residual_connection = nn.ModuleList()
        self.skip_connection = nn.ModuleList()
        verbose = True 

        # dilations repeat num_blocks times
        for i in range(num_blocks):
            
            new_dilation = 1
            additional_scope = kernel_size - 1

            for j in range(num_layers):
                self.dilation.append((new_dilation, init_dilation))
                if verbose: 
                    print(f' ==> dilation block {i}, layer: {j}, dilation: {init_dilation}, new_dilation: {new_dilation}')
    
                self.filter.append(nn.Conv1d(in_channels = residual_channels, out_channels = dilation_channels, 
                                             kernel_size = kernel_size,  bias=bias))
                self.gate.append(nn.Conv1d(in_channels = residual_channels, out_channels = dilation_channels, 
                                             kernel_size = kernel_size,  bias=bias)) 

                # why is the residual connection parametrized? 
                self.residual_connection.append(nn.Conv1d(in_channels = dilation_channels, out_channels = residual_channels, 
                                                    kernel_size = 1,  bias=bias))

                # skip connection is also parametrized 
                self.skip_connection.append(nn.Conv1d(in_channels = dilation_channels, out_channels = skip_channels, 
                                                 kernel_size = 1,  bias=bias))
                #receptive_field  = 
                init_dilation = new_dilation
                new_dilation = new_dilation * 2

                receptive_field += additional_scope
                additional_scope *= 2

        self.end_conv_1 = nn.Conv1d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=1,
                                    bias=True)

        self.end_conv_2 = nn.Conv1d(in_channels=end_channels,
                                    out_channels=classes,
                                    kernel_size=1,
                                    bias=True)

        self.output_length = output_length
        self.receptive_field = receptive_field

    def forward(self, x, generation = False):
       
        x = self.causal_conv(x)
        
        for i in range(self.num_blocks*self.num_layers):
            
            dilation, init_dilation = self.dilation[i]
            # add dilation code here 

            residual = self.wavenet_dilate(x, dilation, init_dilation, i)

            filter = self.filter[i](residual)
            gate = self.gate[i](residual)
            
            filter = F.tanh(filter)
            gate = F.sigmoid(gate)

            x = filter * gate
            s = x

            if x.size(2) != 1:
                s = dilate(x, 1, init_dilation=dilation)
            s = self.skip_connection[i](s)
            try:
                skip = skip[:, :, -s.size(2):]
            except:
                skip = 0
            skip = s + skip

            x = self.residual_connection[i](x)
            # import code; code.interact(local=dict(globals(), **locals()))
            
            x = x + residual[:, :, (self.kernel_size - 1):]


        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        # post process the output --> input shape 
        if not generation:
            [n, c, l] = x.size()

            l = self.output_length
            x = x[:, :, -l:]
            x = x.transpose(1, 2).contiguous()
            x = x.view(n * l, c)

        return x 

    def wavenet_dilate(self, input, dilation, init_dilation, i):
        x = dilate(input, dilation, init_dilation)

        return x
    def generate(self, model, num_samples, temperature = 1, first_samples = None, regularize = 0, ): 
        model.eval() 
        if first_samples is None: 
            first_samples = torch.LongTensor(1).zero_() + (self.classes // 2) 
            first_samples = Variable(first_samples) 
            input = Variable(torch.FloatTensor(1, self.classes, 1).zero_()) 
            input = input.scatter_(1, first_samples[0:1].view(1, -1, 1), 1.) 
            # if first_samples is None: 
            # # first_samples = torch.LongTensor(1).zero_() + (self.classes // 2) 
            # # first_samples = Variable(first_samples) 
        if first_samples is None: 
            first_samples = self.dtype(1).zero_() 
            generated = Variable(first_samples, volatile=True) 
            num_pad = self.receptive_field - generated.size(0) 
        if num_pad > 0: 
            generated = torch.nn.ConstantPad1d(num_pad, 0)(generated) 
            print("pad zero") 
        # generated = torch.tensor(0, device='cuda:0') 
        for i in range(num_samples): 
            input = Variable(torch.FloatTensor(1, self.classes, self.receptive_field).zero_()) 
            if generated.any(): 
                print(f'input {input.size()}, generated {generated.size()}') 
                # self.receptive_field = self.receptive_field.to(torch.int64) 
                input = input.scatter_(1, generated[-self.receptive_field:].view(1, -1, self.receptive_field), 1.) 
            input = input.to('cuda') 
            x = model(input, generation = True)[:, :, -1].squeeze() 
            x = x.detach().to('cpu') 
            if temperature > 0: 
                x /= temperature 
                prob = F.softmax(x, dim=0) 
                print(prob)
                np_prob = np.array(prob) 
                print(np_prob) 
                x = np.random.choice(self.classes, p=np_prob) 
                x = Variable(torch.LongTensor([x])) 
            else: 
                x = torch.max(x, 0)[1].float() 
            if generated.any(): 
                x = x.to(torch.int64) 
                generated = torch.cat((generated, x.reshape(1)), 0) 
            else: 
                generated = x 
                
        generated = (generated / self.classes) * 2. - 1 

        mu_gen = mu_law_expansion(generated, self.classes) 
     
        mu_gen = torch.unsqueeze(mu_gen, 0) 
        ta.save('test_1.wav', mu_gen, 16000) 
        self.train() 

        import code; code.interact(local = locals()) 

        return mu_gen

# model = WaveNet(input_channels = 1, residual_channels = 16, dilation_channels = 16, skip_channels = 16, num_blocks = 2, num_layers = 10)

# sample_tensor =torch.randn(1, 1, 1000)

# model.forward(sample_tensor)