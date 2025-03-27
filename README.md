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
- Bach’s Monumental Chaconne: Johann Sebastian Bach’s chaconne has been arranged for nearly every instrument: from the ominous-sounding organ to the solo flute or the delightfully sparse marimba. Or, listen to these renditions by an enterprising clarinetist and an equally ambitious saxophonist. Or how about this menacing arrangement for trumpet and orchestra and this lyrical performance by two cellos. There is truly an arrangement for everyone. (dataset was downsamples, and quantized further to make it suitable for nn training)


### Conditional generation augmentation 

### Hyperparameters 

1. Epochs: 10 
2. Learning Rate, Scheduler: 0.001, None 
3. Optimizer: Adam
4. Criteria: CrossEntropy
5. Grad clip and weight decay: 0
            