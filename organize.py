# from faker import Faker

# fake = Faker()
# for _ in range(10):
#     print(fake.file_name())

import torch
from torch.multiprocessing import Pool

def run_program(xx):
    print(f'running {xx}')

    import torch
    net = torch.nn.Linear(1000, 1000)
    opt = torch.optim.Adam(net.parameters())

    for i in range(500):
        x = torch.randn(1000)
        y = torch.randn(1000)
        yp = net(x)
        loss = (yp - y).pow(2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f'done {xx}')
    return 4


from datetime import datetime

import numpy as np

if __name__ == '__main__':
    print('hello!!!')
    # import sys
    # t = int(sys.argv[-1])
    # start = datetime.now()
    # with Pool(t) as p:
    #     print(p.map(run_program, np.arange(12)))
    # end = datetime.now()
    # print(end-start)

