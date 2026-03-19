import numpy as np

class Estimator:

    def __init__(self, gc_scales, pos_0, Nx=20, Ny=20):
        self.Nx = Nx
        self.Ny = Ny
        self.gc_scales = gc_scales
        self.cell_dist = self.get_distances()
        self.estimations = []
        for i_mod in range(len(gc_scales)):
            self.estimations.append([pos_0])

    def e_dist(self, cell):
        return np.sqrt(np.sum(np.square(cell)))
    def get_cell_dist(self, c_r, c_c):
        dist_mat = np.zeros((self.Ny, self.Nx, 2))
        for row in range(-self.Ny, self.Ny):
            for col in range(-self.Nx, self.Nx):
                roww = c_r + row
                coll = c_c + col
                if roww < 0 or roww >= self.Ny:
                    roww = roww % self.Ny
                    coll = (coll + 10) % self.Nx
                if coll < 0 or coll >= self.Nx:
                    coll = coll % self.Nx
                if dist_mat[roww][coll][0] == 0 and dist_mat[roww][coll][1] == 0:
                    dist_mat[roww][coll][0] = col
                    dist_mat[roww][coll][1] = row
                elif self.e_dist(dist_mat[roww][coll]) > self.e_dist([row * np.sqrt(3) / 2, col]):
                    dist_mat[roww][coll][0] = col
                    dist_mat[roww][coll][1] = row

        dist_mat[c_r][c_c][0] = 0
        dist_mat[c_r][c_c][1] = 0
        return dist_mat

    def get_distances(self):
        oo = self.get_cell_dist(0, 0)
        cell_distances = []

        for s_row in range(self.Ny):
            aux = []
            for s_col in range(self.Nx):
                x = np.roll(oo, s_col, axis=1)
                x = np.roll(x, s_row, axis=0)
                y = np.roll(x[0:s_row], int(self.Nx / 2), axis=1)
                x[0:s_row] = y
                aux.append(x)
            cell_distances.append(aux.copy())
        cell_distances = np.array(cell_distances)
        return cell_distances

    def new_estimation(self, start_grid_state, final_grid_state):

        mean_estimated_x = 0
        mean_estimated_y = 0
        mod_estimations = []
        for i_mod in range(len(self.gc_scales)):

            start_center = np.unravel_index(np.argmax(start_grid_state[:, :, i_mod]), (self.Nx, self.Ny))
            final_center = np.unravel_index(np.argmax(final_grid_state[:, :, i_mod]), (self.Nx, self.Ny))

            delta = self.cell_dist[final_center[0], final_center[1], start_center[0], start_center[1]]
            new_pos_x = self.estimations[i_mod][-1][0] + (1 - self.gc_scales[i_mod]) * delta[0]
            new_pos_y = self.estimations[i_mod][-1][1] + (1 - self.gc_scales[i_mod]) * delta[1]
            self.estimations[i_mod].append((new_pos_x, new_pos_y))
            
            mod_estimations.append([new_pos_x, new_pos_y])
            mean_estimated_x += new_pos_x
            mean_estimated_y += new_pos_y
        
        mean_estimated_x = mean_estimated_x/len(self.gc_scales)
        mean_estimated_y = mean_estimated_y/len(self.gc_scales)
        return mean_estimated_x, mean_estimated_y, mod_estimations
      

        