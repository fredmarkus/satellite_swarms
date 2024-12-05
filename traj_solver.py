import numpy as np
import jax
import jax.numpy as jnp

from sat_dynamics import rk4_discretization

class trajSolver:
    def __init__(self, x_traj, y_m, sat, N, meas_dim, bearing_dim, n_sats, MU, state_dim, dt):
        self.x_traj = x_traj
        self.y_m = y_m
        self.sat = sat
        self.N = N
        self.meas_dim = meas_dim
        self.bearing_dim = bearing_dim
        self.n_sats = n_sats
        self.MU = MU
        self.state_dim = state_dim
        self.dt = dt
        self.cov = self.sat.R_weight*np.eye(self.meas_dim)
        self.inv_cov = np.linalg.inv(self.cov)

    def objective(self, x):
        obj = 0
        for i in range(self.N):
            start_i = i*6
            start_i1 = (i+1)*6
            if self.y_m[i,0:self.bearing_dim,self.sat.id].any() != 0:
                obj += (self.y_m[i,0:self.bearing_dim,self.sat.id] - self.sat.h_landmark(x[start_i:start_i+3]).T)@self.inv_cov@(self.y_m[i,0:self.bearing_dim,self.sat.id] - self.sat.h_landmark(x[start_i:start_i+3]))
            # add the dynamics to the objective
            if i < self.N-1: # Don't add the dynamics optimization for the last time step
                obj += (x[start_i1:start_i1+6] - rk4_discretization(x[start_i:start_i+6], self.dt))@(x[start_i1:start_i1+6] - rk4_discretization(x[start_i:start_i+6], self.dt))
            for j in range(self.bearing_dim,self.meas_dim):
                # print(j, y_m[i,j,self.sat.id])
                if self.y_m[i,self.bearing_dim:self.meas_dim,self.sat.id].any() != 0:
                    obj += (1/self.sat.R_weight)*(self.y_m[i,j,self.sat.id] - self.sat.h_inter_range(i, j, x[start_i:start_i+3]))**2
        return obj
    

    def gradient(self, x):
        grad = jax.grad(self.objective)(x)

        return grad

    def constraints(self, x):
        
        g = jnp.zeros((self.N)*self.state_dim)
        # Position initial conditions
        g = g.at[0].set(x[0] - self.sat.x_0[0]) 
        g = g.at[1].set(x[1] - self.sat.x_0[1])
        g = g.at[2].set(x[2] - self.sat.x_0[2])
        # Velocity initial conditions
        g = g.at[3].set(x[3] - self.sat.x_0[3])
        g = g.at[4].set(x[4] - self.sat.x_0[4])
        g = g.at[5].set(x[5] - self.sat.x_0[5])


        for i in range(self.N-1):
            x_i = x[i*self.state_dim:(i+1)*self.state_dim]
            x_ip1 = x[(i+1)*self.state_dim:(i+2)*self.state_dim]
            
            x_new = rk4_discretization(x_i, self.dt)
            g = g.at[(i+1)*self.state_dim:(i+2)*self.state_dim].set(x_new - x_ip1)

        return g

    def jacobian(self, x):
        jacobian = jax.jacfwd(self.constraints)(x)
        jacobian = jacobian[self.jrow, self.jcol]
        
        return jacobian

    
    def jacobianstructure(self):
        dim = 6
        row = []
        col = []

        # Initial conditions
        row.extend([0,1,2,3,4,5])
        col.extend([0,1,2,3,4,5])

        # Define offsets for the row and col calculations
        offsets = [
            (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
            (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 7),
            (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 8),
            (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 9),
            (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 10),
            (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 11)
        ]

        # Loop over the range and apply the offsets
        for i in range(0, self.N-1):
            for row_offset, col_offset in offsets:
                row.append((i+1) * dim + row_offset)
                col.append(i * dim + col_offset)
    
        # Visualization of the Jacobian mask. Check it matches the actual Jacobian. Using '11' rather than '1' for easier intepretation
        # jac = np.zeros((self.N*6,self.N*6))
        # for i in range(len(row)):
        #     jac[row[i],col[i]] = 11

        # print(jac)
        self.jrow = np.array(row, dtype=int)
        self.jcol = np.array(col, dtype=int)

        return np.array(row, dtype=int), np.array(col, dtype=int)

    # Currently we are still relying on the default hessian approximation which is ok but leads to performance deficits

    ### NOTE: HESSIAN IS ONLY CALLED WHEN HESSIAN_APPROXIMATION IS SET TO 'EXACT' ###
    def hessian(self, x, lagrange, obj_factor):
        hess = obj_factor*jax.hessian(self.objective)(x)
        hess3d = jax.hessian(self.constraints)(x)
        hess2 = np.tensordot(hess3d, lagrange, axes=([2], [0]))
        hess_final = hess + hess2
        hess_final = hess_final[self.hrow, self.hcol]
        return hess_final

    def hessianstructure(self):
        half_dim = 3
        row = []
        col = []

        offsets = [
            (0, 0), 
            (1, 0), (1, 1),
            (2, 0), (2, 1), (2, 2)                
        ]

        for i in range(0, self.N):
            for row_offset, col_offset in offsets:
                row.append(i * half_dim + row_offset)
                col.append(i * half_dim + col_offset)
    
        hess = np.zeros((self.N*6,self.N*6))
        for i in range(len(row)):
            hess[row[i],col[i]] = 11

        # print(hess)
        self.hrow = np.array(row, dtype=int)
        self.hcol = np.array(col, dtype=int)

        return np.array(row, dtype=int), np.array(col, dtype=int)
