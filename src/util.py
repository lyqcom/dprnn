import numpy as np

def check_parameters(net):
    '''
        Returns module parameters. Mb
    '''
    # parameters = sum(param.numel() for param in net.get_parameters())
    # return parameters / 10**6
    total_params = 0

    for param in net.trainable_params():
        total_params += np.prod(param.shape)

    return total_params / 10**6

