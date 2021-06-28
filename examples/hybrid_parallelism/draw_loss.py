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


c1, c2 = None, None
cc = 0
clen = 0
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
    if cc == 0:
        c1 = cost
    else:
        c2 = cost
    cc+=1
    # plt.figure('align loss')
    # plt.plot(step, loss)
    plt.figure('align lr')
    plt.plot(step, lrs)
clen = min(len(c1), len(c2))
c1 = c1[:clen]
c2 = c2[:clen]
plt.figure("diff")
plt.plot(range(0,clen), [ l1 - l2 for (l1, l2) in zip(c1, c2) ])
plt.figure("align cost")
plt.savefig("align_cost.png")
plt.figure("align lr")
plt.savefig("align_lr.png")
plt.show()

