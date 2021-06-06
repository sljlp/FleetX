from matplotlib import pyplot as plt
import sys
# plt.figure('0')
# plt.figure('1')
# plt.figure('2')
# plt.figure(3)

def parse_line(line):
    '''step: 1618, cost: 7.040180, mlm loss: 7.040180, sentence order acc:'''
    step_idx, cost_idx, loss_idx, sen_idx = line.find('step:'), line.find(', cost:'), line.find(', mlm loss:'), line.find(', sentence order acc:')
    '''learning rate: 1.617e-05, loss_scalings: '''
    lr_idx, lsscale_idx = line.find('learning rate: '), line.find(', loss_scalings:')
    step = int(line[step_idx + 5 : cost_idx])
    cost = float(line[cost_idx + 7 : loss_idx])
    loss = float(line[loss_idx + 11 : sen_idx])
    lr = float(line[lr_idx + 14 : lsscale_idx])
    return step, cost, loss, lr

for name in sys.argv[1:]:
    step, cost, loss, lrs = [], [], [], [] 
    for line in open(name):
        if 'step' in line and 'cost' in line and 'loss' in line and 'learning rate' in line:
            s, c, l, lr = parse_line(line)
            step.append(s)
            cost.append(c)
            loss.append(l)
            lrs.append(lr)
    plt.figure('align cost')
    plt.plot(step, cost)
    # plt.figure('align loss')
    # plt.plot(step, loss)
    plt.figure('align lr')
    plt.plot(step, lrs)
plt.show()

