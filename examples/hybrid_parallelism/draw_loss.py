from matplotlib import pyplot as plt
import sys
# plt.figure('0')
# plt.figure('1')
# plt.figure('2')
# plt.figure(3)

def parse_line(line):
    '''step: 1618, cost: 7.040180, mlm loss: 7.040180, sentence order acc:'''
    last_key = ', sentence order acc:' if ', sentence order acc:' in line else ', speed:'
    
    step_idx, cost_idx, loss_idx, sen_idx = line.find('step:'), line.find(', cost:'), line.find(', mlm loss:'), line.find(last_key)
    '''learning rate: 1.617e-05, loss_scalings: '''
    lr_idx, lsscale_idx, pploss_idx = line.find('learning rate: '), line.find(', loss_scalings:'), line.find(", pp_loss:")
    step = int(line[step_idx + 5 : cost_idx])
    cost = float(line[cost_idx + 7 : loss_idx])
    loss = float(line[loss_idx + 11 : sen_idx])
    lr = float(line[lr_idx + 14 : lsscale_idx])
    loss_scaling = float(line[lsscale_idx + 16: pploss_idx])
    # print(loss_scaling)
    return step, cost, loss, lr, loss_scaling


c1, c2 = None, None
cc = 0
clen = 0
for name in sys.argv[1:]:
    step, cost, loss, lrs, ls_scale = [], [], [], [], [] 
    for line in open(name):
        if 'step' in line and 'cost' in line and 'loss' in line and 'learning rate' in line:
            s, c, l, lr, lss = parse_line(line)
            step.append(s)
            cost.append(c)
            loss.append(l)
            lrs.append(lr)
            # print(lss)
            ls_scale.append(lss)
    plt.figure('align cost')
    plt.plot(step, cost, label=name)
    plt.legend(loc=0,)
    if cc == 0:
        c1 = cost
    else:
        c2 = cost
    cc+=1
    # plt.figure('align loss')
    # plt.plot(step, loss)
    plt.figure('align lr')
    plt.plot(step, lrs, label=name)
    plt.legend(loc=0,)

    plt.figure("ls scaling")
    # print(ls_scale)
    plt.plot(step, ls_scale, label=name)
    plt.legend(loc=0,)

clen = min(len(c1), len(c2))
c1 = c1[:clen]
c2 = c2[:clen]
plt.figure("diff")
plt.plot(range(0,clen), [ l1 - l2 for (l1, l2) in zip(c1, c2) ])
plt.plot(range(clen), [0 for _ in range(clen)])
plt.figure("align cost")
plt.savefig("align_cost.png")
plt.figure("align lr")
plt.savefig("align_lr.png")

plt.show()

