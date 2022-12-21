import mindspore.nn as nn
import mindspore.ops as ops

class TrainOneStep(nn.TrainOneStepCell):
    def __init__(self , network , optimizer , sens=1.0):
        super(TrainOneStep, self).__init__(network, optimizer, sens)
        self.network = network
        

    def construct(self, padded_mixture, mixture_lengths, padded_source):

        loss = self.network(padded_mixture, mixture_lengths, padded_source)

        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)

        grads = self.grad(self.network, self.weights)(padded_mixture, mixture_lengths, padded_source, sens)

        grads = self.grad_reducer(grads)

        loss = ops.depend(loss, self.optimizer(grads))
        return loss