# Bidirectional Backpropagation Network - Summary

In deep neural networks (DNN), backpropagation (BP) method is considered to be one of the most important algorithms to build a successful model. However, it has been
suggested that in a biological neural system, it is not possible to build a backward channel that is connected symmetrically to the corresponding feedforward weight. Furthermore, there is no solid evidence indicating that errors can be backpropagated in the brain. As a result, it will prevent the backward algorithm to obtain the value of the feedforward weight, which is crucial in calculating the weight gradients.

To adjust the connection weight without any prior information of the feedforward phase, two models are introduced: the feedback-alignment (FA) model and the direct feedback-
alignment (DFA) model. The basic idea about the FA model is that it removes the assumption of knowing the connection weight and propagate some random feedback from the output
layer to the hidden layers in order to update the weight gradients. The difference between DFA and FA is that in DFA, the feedback error signal can transmit directly from the output layer to any of the hidden layers.

However, since FA and DFA models send random and fixed feedback, their performance is not optimized. Hence, two high performance bidirectional neural network models are proposed. They discard the assumption that the feedback weights are random and fixed. In particular, each of these two models consists of two weight matrices: the feed forward and the feedback weight matrices, where the former weight matrix is tuned during the forward learning phase, and the latter weight matrix is adjusted in the backward learning
phase.

The details of the implementation of the bidirectional network was discussed. In particular, we showed the difference in the calculation of the derivatives between each type of Neural Networks, as well as providing detailed explanation of the creation of new functions to build the bidirectional neural network.


## Experiment
We performed our expriment with the MNIST dataset. All networks discussed above are trained using 50,000 training samples from MINIST dataset, and there are 10,000 testing samples to evaluate their classification accuracy. Each network is trained 100 epochs with learning rate 0.01 and batch size 128. The activation functions for the output layer and hidden
layers are Softmax and Tanh, respectively. Furthermore, we use Categorical Cross-Entropy as our loss function. Below is the performance:

![alt text](https://github.com/xpada001/bidirect_backprop/blob/main/MNIST_perf.png?raw=true)

The BDFA model gives us the most accurate results on MNIST. The error rates of FA and DFA are very close and are significantly higher than the error rates of BFA and BDFA. This is an evidence that trainable feedback weights would be preferred to random feedback weights during the training process. However, there is a time trade off for high accuracy.

### Flipped BFA and BDFA
In traditional BFA and BDFA, FeedForward and BackProp functions are called before FeedBack and BackPropFeedBack. We would like to determine if the classification accuracy of
bidirectional models would be affected if functions are called in different orders. Two new models called ”Flipped BFA” and ”Flipped BDFA” are created, which always feed the data in before backward propagation. In other words, the order of functions calls in our new models becomes FeedForward, FeedBack, BackProp and BackPropFeedBack. We run the experiments using MNIST dataset with the exactly same settings as in the last section, and the results are shown in the report. We can observe that ”Flipped” models have slightly lower accuracy and longer running time compared to traditional BFA and BDFA models. Hence, finding an appropriate order of function calls during the training phase is important to achieve a good model performance.

### Regression
Besides classification, we would like to apply our BFA and BDFA models on other types of problems. For instance, we would like to train them on regression-type dataset on both 2
dimensions and 3 dimensions.

### Adversarial Input
Based on the experiments conducted above, one could think that the bidirectional models are supposed to have better weight matrices. Therefore, it would be interesting to
test its robustness against targeted and untargeted adversarial inputs.

## More information

For more information, including visualization of the mentioned neural networks, it can be found in **report.pdf**. 

### Note
The jupyter notebook is missing a part of the code that I can no longer retrieve. For any questions, I would love to discuss further by sending me an email at chensiao06@gmail.com
