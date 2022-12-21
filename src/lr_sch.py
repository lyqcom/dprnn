

def dynamic_lr(step_per_epoch, epoch):
    lr = []

    for ep in range(1, epoch + 1):
        for step_num in range(1, step_per_epoch + 1):
            if step_num <= 4000:
                temp = 0.2 * ( 64 ** (-0.5)) * min(step_num ** (-0.5), step_num * (4000 ** (-1.5)))
                lr.append(temp)
            else:
                temp = 0.0004 * (0.98 ** (ep // 2))
                lr.append(temp)
    return lr