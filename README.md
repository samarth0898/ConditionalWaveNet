## A minimum implementation of WaveNet for audio modelling and generation 


### Implementation of WaveNet, an autoregressive generative model 
            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

1. Causal Conv (1x1)
2. Residual Block x num_blocks
    - Dilated Conv
    - Residual connections 
        - tanh * sigmoid 
        - Casual Conv (1x1) --> Skip connection
        - + Residual connection 
3. Skip connection 



            