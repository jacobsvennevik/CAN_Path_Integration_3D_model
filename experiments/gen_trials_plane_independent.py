import numpy as np
import glob

from grid_cells.path_integration import PathIntegrator

change_ints = (1, 5, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5)

pi = PathIntegrator()

for k, xfname in enumerate(glob.glob('data/x/*.npy')):
    tid = xfname[7:15]
    x = np.load(xfname)
    for kappa in (500, 400, 300):
        for change_int in change_ints:
            T = len(x)
            u, r, phi, b = pi.calc_u(x, change_int=int(change_int), linear=True)

            if kappa == 500:
                a = pi.sim(u, r, phi, b, B_rotz=8, change_int=int(change_int), uncert=False)
                np.save('data/linear/a_c/c_' + tid + '_' + str(int(change_int)), a)

            a = pi.sim(u, r, phi, b, B_rotz=8, change_int=int(change_int), kappa=float(kappa), uncert=True)
            np.save('data/linear/a_%d/u_' % kappa + tid + '_' + str(int(change_int)), a)
