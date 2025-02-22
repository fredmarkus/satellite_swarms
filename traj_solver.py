import numpy as np
import jax
import jax.numpy as jnp

from sat.dynamics import f_j

class trajSolver_v2:
    def __init__(self, x_traj, N, n_sats, state_dim, dt, is_initial, sats_collection, meas_type):
        self.sat_collection = sats_collection
        self.x_traj = x_traj
        self.sat = self.sat_collection[0] # Use satellite 0
        self.N = N
        self.n_sats = n_sats
        self.state_dim = state_dim
        self.dt = dt
        self.is_initial = is_initial
        self.meas_type = meas_type

    def objective(self, x):
        obj = 0

        for i in range(self.N):

            for sat in self.sat_collection:
                sat.curr_pos = self.x_traj[i + 1, 0:3, sat.id]

            # Get visible landmarks and satellites using actual current position
            if "land" in self.meas_type:
                visible_landmarks = self.sat.visible_landmarks_list()
                self.sat.land_bearing_dim = len(visible_landmarks) * 3
 
            visible_sats = self.sat.visible_sats_list(self.sat_collection)

            if "sat_bearing" in self.meas_type:
                self.sat.sat_bearing_dim = len(visible_sats) * 3

            if "range" in self.meas_type:
                self.sat.range_dim = len(visible_sats)
            
            meas_dim = self.sat.land_bearing_dim + self.sat.sat_bearing_dim + self.sat.range_dim
            
            if i < self.N - 1:
                term = x[(i+1)*6:(i+2)*6] - f_j(x[i*6:(i+1)*6],self.dt)
                obj += term.T@term

            if meas_dim > 0:
                start_i = i*self.state_dim
                
                if self.sat.land_bearing_dim > 0:
                    y_m = self.sat.measure_z_landmark()
                    inv_cov = np.linalg.inv(self.sat.R_weight_land_bearing*np.eye(self.sat.land_bearing_dim))
                    obj += (y_m - self.sat.h_landmark(x[start_i:start_i + 3])).T@inv_cov@(y_m - self.sat.h_landmark(x[start_i:start_i+3]))
                
                if self.sat.sat_bearing_dim > 0:
                    y_m = self.sat.measure_z_sat_bearing()
                    inv_cov = np.linalg.inv(self.sat.R_weight_sat_bearing*np.eye(self.sat.sat_bearing_dim))
                    obj += (y_m - self.sat.h_sat_bearing(x[start_i:start_i + 3])).T@inv_cov@(y_m - self.sat.h_sat_bearing(x[start_i:start_i+3]))

                if self.sat.range_dim > 0:
                    y_m = self.sat.measure_z_range()
                    inv_cov = np.linalg.inv(self.sat.R_weight_range*np.eye(self.sat.range_dim))
                    obj += (y_m - self.sat.h_inter_range(x[start_i:start_i + 3])).T@inv_cov@(y_m - self.sat.h_inter_range(x[start_i:start_i+3]))

        return obj
    

    def gradient(self, x):
        grad = jax.grad(self.objective)(x)

        return grad

    def constraints(self, x):
        if not self.is_initial:
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
                
                x_new = f_j(x_i, self.dt)
                g = g.at[(i+1)*self.state_dim:(i+2)*self.state_dim].set(x_new - x_ip1)
            
        else:
            g = jnp.zeros((self.N-1)*self.state_dim)

            for i in range(self.N-1):
                x_i = x[i*self.state_dim:(i+1)*self.state_dim]
                x_ip1 = x[(i+1)*self.state_dim:(i+2)*self.state_dim]
                
                x_new = f_j(x_i, self.dt)
                g = g.at[(i)*self.state_dim:(i+1)*self.state_dim].set(x_new - x_ip1)

        return g

    def jacobian(self, x):
        jacobian = jax.jacfwd(self.constraints)(x)
        # print(jacobian.T)
        jacobian = jacobian[self.jrow, self.jcol]
        
        return jacobian

    
    def jacobianstructure(self):
        dim = 6
        row = []
        col = []

        # Initial conditions (only added if we are not in the initial case)
        if not self.is_initial:
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
        if not self.is_initial:
            for i in range(0, self.N-1):
                for row_offset, col_offset in offsets:
                    row.append((i+1) * dim + row_offset)
                    col.append(i * dim + col_offset)
    
        else:
            for i in range(0, self.N-1):
                for row_offset, col_offset in offsets:
                    row.append(i * dim + row_offset)
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
