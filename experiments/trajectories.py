import numpy as np

from grid_cells.path_integration import PathIntegrator


def trajectory3d(T, vi_max, x_init=None, xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1)):
    x = np.zeros((T, 3))
    if x_init is not None:
        x[0] = x_init

    for t in range(1, T):
        for i, (lb, ub) in enumerate((xlim, ylim, zlim)):
            lb_ = x[t - 1, i] - vi_max if x[t - 1, i] - vi_max > lb else lb + 1e-5
            ub_ = x[t - 1, i] + vi_max if x[t - 1, i] + vi_max < ub else ub
            x[t, i] = lb_ + np.random.rand() * (ub_ - lb_)
    return x


N = 8  # How many trials you want
T = 100000

for i in range(N):
    x = trajectory3d(T=T, vi_max=0.08)
    np.save('data/x/%i.npy' % i, x)
