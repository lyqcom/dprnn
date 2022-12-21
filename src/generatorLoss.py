import mindspore.nn  as nn
from network_define import WithLossCell
from Loss_final1 import loss

class Generatorloss(nn.Cell):
    def __init__(self , generator):
        super(Generatorloss, self).__init__()
        # 下面定义需要用到的loss
        self.generator = generator
        self.my_loss = loss()
        self.net_with_loss =WithLossCell(self.generator, self.my_loss)

    def construct(self, mixture, len, source):
        # 下面是计算loss的流程
        # prediction = self.generator(maxture)
        myLoss = self.net_with_loss(mixture, len, source)
        return myLoss



