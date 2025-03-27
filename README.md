## A minimum implementation of WaveNet for audio modelling and generation 


### Structure of WaveNet block

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*



### Dataset 

### Hyperparameters 

1. Epochs: 10 
2. Learning Rate, Scheduler: 0.001, None 
3. Optimizer: Adam
4. Criteria: CrossEntropy
5. Grad clip and weight decay: 0
            