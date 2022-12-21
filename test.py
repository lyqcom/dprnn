import numpy as np
from mindspore import Tensor
import mindspore.nn as nn

net = nn.LSTM(10, 16, 2, has_bias=True, batch_first=True, bidirectional=False)
x = Tensor(np.ones([3, 5, 10]).astype(np.float32))
h0 = Tensor(np.ones([1 * 2, 3, 16]).astype(np.float32))
c0 = Tensor(np.ones([1 * 2, 3, 16]).astype(np.float32))
output, (hn, cn) = net(x, (h0, c0))
print(output.shape)