#!/usr/bin/env python3

"""
This file contains the LoopClosureAwareMPC class, which is used to solve Active SLAM problems.
"""

__author__ = "Kota Kondo"
__email__ = "kkondo@mit.edu"
__date__ = "2023-12-22"
__version__ = "0.0.1"

import numpy as np
import casadi
from casadi import Opti


class LoopClosureAwareMPC():
    """
    This class is used to formulate loop closure aware MPC problems.

    Attributes:
        dynamics: Dynamics object
        occupancy_map: OccupancyMap object
        num_agents: Number of agents
        num_timesteps: Number of timesteps
        min_allowable_dist: Minimum allowable distance between agents
        x_bounds: Bounds on states
        u_bounds: Bounds on inputs
    """

    def __init__(self, dynamics, occupancy_map, num_agents=1, num_timesteps=100, 
                    min_allowable_dist=0.5, x_bounds=None, u_bounds=None,
                    terminal_cost_weight=1.0, waypoints_cost_weight=1.0, input_cost_weight=1.0,
                    travel_cost_weight=1.0, travel_dist_cost_weight=1.0, input_rate_cost_weight=1.0,
                    collision_cost_weight=1.0):

        self.solver_opts = {"ipopt.tol":1e-3, "expand":True,
                            'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes', 
                            'ipopt.max_iter': 3000, 'ipopt.linear_solver': 'ma97'}

        # problem parameters
        self.dynamics = dynamics
        self.occupancy_map = occupancy_map
        self.M = num_agents
        self.N = num_timesteps
        self.min_allowable_dist = min_allowable_dist
        self.x_bounds = x_bounds
        self.u_bounds = u_bounds 

        # cost weights
        self.terminal_cost_weight = terminal_cost_weight
        self.waypoints_cost_weight = waypoints_cost_weight
        self.input_cost_weight = input_cost_weight
        self.travel_cost_weight = travel_cost_weight
        self.travel_dist_cost_weight = travel_dist_cost_weight
        self.input_rate_cost_weight = input_rate_cost_weight
        self.collision_cost_weight = collision_cost_weight

    def solve_opt(self, x_guess=None, u_guess=None):
        '''
        Solve optimization problem.
        
        Returns: 
            (M x n x N+1) states at the beginning of each timestep
            (M x m x N) control to be applied at the beginning of each timestep
            (N+1) times
        '''

        # you cannot just assgin the initial guess to the opti variables as a whole
        # so we have to loop through and set the initial guess for each variable      
        if x_guess is not None:
            for m in range(self.M):
                for i in range(self.x0.shape[0]):
                    for j in range(self.N+1):
                        self.opti.set_initial(self.x[m][i, j], x_guess[i, j])
        if u_guess is not None:
            for m in range(self.M):
                for i in range(self.dynamics.u_shape):
                    for j in range(self.N):
                        self.opti.set_initial(self.u[m][i, j], u_guess[i,j])

        self.opti.solver('ipopt', self.solver_opts)
        sol = self.opti.solve()
        self.x_sol = [sol.value(self.x[m]) for m in range(self.M)]
        self.u_sol = [sol.value(self.u[m]) for m in range(self.M)]
        self.tf_sol = sol.value(self.tf)
        # get cost
        cost = sol.value(self.opti.f)
        return self.x_sol, self.u_sol, np.linspace(0.0, self.tf_sol, self.N+1), cost

    def setup_mpc_opt(self, x0, xf, tf, waypoints={}, Qf=None, R=None, x_bounds=None, u_bounds=None):
        '''
        x0: nx1 initial state
        xf: nx1 goal state
        tf: end time
        Qf: nxn weighting matrix for xf cost
        '''
        self._general_opt_setup()
        self.x0 = x0
        self.xf = xf
        self.tf = tf # this can be a variable self.tf = self.opti.variable() (this didn't work well tho)
        if Qf is None:
            Qf = np.eye(self.dynamics.x_shape)
        self.dt = self.tf / self.N

        # initialize costs
        terminal_cost = 0.
        waypoints_cost = 0.
        input_cost = 0.
        travel_cost = 0.
        travel_dist_cost = 0.
        input_rate_cost = 0.
        collision_cost = 0.

        # add costs
        for m in range(self.M):

            # terminal cost
            terminal_cost = self._add_terminal_cost(Qf, m)

            # waypoints cost
            waypoints_cost = self._add_waypoints_cost(waypoints, m)
            
            # input cost
            if R is not None:
                input_cost = self._add_input_cost(R, m)

            # travel time cost
            travel_cost = self.tf

            # travel distance cost
            travel_dist_cost = self._add_travel_dist_cost(m)

            # input rate cost
            input_rate_cost = self._add_input_rate_cost(m)
            
            # collision cost
            collision_cost = self._add_collision_cost(m)

        # total cost
        mpc_cost = self.terminal_cost_weight * terminal_cost + \
                    self.waypoints_cost_weight * waypoints_cost + \
                    self.input_cost_weight * input_cost + \
                    self.travel_cost_weight * travel_cost + \
                    self.travel_dist_cost_weight * travel_dist_cost + \
                    self.input_rate_cost_weight * input_rate_cost + \
                    self.collision_cost_weight * collision_cost

        # minimize cost
        self.opti.minimize(mpc_cost)

        # constraints
        self._add_dynamic_constraints()
        self._add_state_constraint(0, x0)
        self._add_state_constraint(-1, xf)
        self.add_x_bounds(x_bounds)
        self.add_u_bounds(u_bounds)
        # self._add_obstacle_constraints()
        # self._add_multi_agent_collision_constraints()

    def add_x_bounds(self, x_bounds):
        ''' 
        x_bounds: n x 2 vector of x min and max (can be infinity)
        Constrain x to stay within x_bounds. 
        '''
        if x_bounds is None:
            return
        for i in range(self.dynamics.x_shape):
            if np.isinf(x_bounds[i,0]) and np.isinf(x_bounds[i,1]):
                continue
            for m in range(self.M):
                self.opti.subject_to(self.opti.bounded(x_bounds[i,0], self.x[m][i,:], x_bounds[i,1]))

    def add_u_bounds(self, u_bounds):
        ''' 
        u_bounds: m x 2 vector of u min and max (can be infinity)
        Constrain u to stay within u_bounds. 
        '''
        if u_bounds is None:
            return
        for i in range(self.dynamics.u_shape):
            for m in range(self.M):
                self.opti.subject_to(self.opti.bounded(u_bounds[i,0], self.u[m][i,:], u_bounds[i,1]))

    def add_u_diff_bounds(self, u_diff_bounds):
        ''' 
        u_diff_bounds: m x 2 vector of u min and max (can be infinity)
        Constrain difference of u to not excede this rate.
        Not fully tested.
        '''
        for i in range(self.dynamics.u_shape):
            if np.isinf(u_diff_bounds[i]):
                continue
            for m in range(self.M):
                for k in range(self.N - 1):
                    self.opti.subject_to(((self.u[m][i,k] - self.u[m][i,k+1])/self.dt)**2 <= (u_diff_bounds[i])**2)

    def add_u0_constraint(self, u0):
        ''' u0 = len n vector. Add constraint at k=0 on input. '''
        self._add_input_constraint(0, u0)

    def add_uf_constraint(self, uf):
        ''' uf = len n vector. Add constraint at final timestep on input. '''
        self._add_input_constraint(-1, uf)

    def draw_path(self, fig=None, ax=None):
        import matplotlib.pyplot as plt
        
        if ax == None:
            fig, ax = plt.subplots()
        colors = ['green', 'blue', 'red', 'orange', 'pink']

        for i in range(self.M):
            ax.plot(self.x_sol[i][self.dynamics.physical_state_idx[0],:], 
                    self.x_sol[i][self.dynamics.physical_state_idx[1],:],
                    color=colors[i])

        ax.set_aspect('equal')
        plural = 's' if self.M > 1 else ''
        ax.set_title(f'{self.M} agent{plural}, {np.round(self.tf_sol, 2)} seconds')

        return fig, ax
    
    def _add_terminal_cost(self, Qf, m):
        """ Add terminal cost """
        return (self.x[m][:,-1] - self.xf.reshape((-1,1))).T @ Qf @ (self.x[m][:,-1] - self.xf.reshape((-1,1)))

    def _add_waypoints_cost(self, waypoints, m):
        """ Add waypoints cost """
        waypoints_cost = 0.
        for i, x in waypoints.items():
            waypoints_cost += (self.x[m][:,i] - x.reshape((-1, 1))).T @ (self.x[m][:,i] - x.reshape((-1,1)))
        return waypoints_cost

    def _add_input_cost(self, R, m):
        """ Add input cost """
        input_cost = 0.
        for i in range(self.N):
            input_cost += self.u[m][:,i].T @ R/self.N @ self.u[m][:,i]
        return input_cost

    def _add_travel_dist_cost(self, m):
        """ Add travel distance cost """
        travel_dist_cost = 0.
        for i in range(self.N):
            travel_dist_cost += (self.x[m][0,i+1] - self.x[m][0,i])**2 + (self.x[m][1,i+1] - self.x[m][1,i])**2
        return travel_dist_cost
    
    def _add_input_rate_cost(self, m):
        """ Add input rate cost """
        input_rate_cost = 0.
        for i in range(self.N - 1):
            input_rate_cost += casadi.norm_2(self.u[m][:,i+1] - self.u[m][:,i])**2
        return input_rate_cost

    def _add_collision_cost(self, m):
        """ Add collision cost """
        collision_cost = 0.

        """ *** option 1: sum up the map cost at each state (doesn't work because of casadi related error) *** """
        # this didn't work due to casadi related error
        # casadi does not support indexing np arrays with casadi variables so we have to convert the map to a casadi parameter
        # costmap = self.opti.parameter(self.occupancy_map.costmap.shape[0], self.occupancy_map.costmap.shape[1])
        # self.opti.set_value(costmap, self.occupancy_map.costmap)
        # for j in range(self.x[m].shape[1]):
        #     # print("careful with this map indexing")
        #     collision_cost += costmap[casadi.floor(self.x[m][0,j]), casadi.floor(self.x[m][1,j])]

        """ *** option 2: occupancy map cost weighted by inverse squared distance (don't forget to add 0.01 to avoid division by zero) *** """
        # this works but very slow
        # (TODO) we can pass just local map to this function instead of the whole occupancy map 
        # for i in range(self.occupancy_map.costmap.shape[0]):
        #     for j in range(self.occupancy_map.costmap.shape[1]):
        #         for k in range(self.x[m].shape[1]):
        #             # print("careful with this map indexing")
        #             collision_cost += self.occupancy_map.costmap[i,j] * ( 1 / ((self.x[m][0,k] - i)**2 + (self.x[m][1,k] - j)**2+0.01) )

        """ *** option 3: create obstacle constraints using occupancy map *** """
        # this works but only accounts for obstacles that have more cost than the threshold
        # create obstacle constraints using occupancy map
        self.obstacles = []
        for i in range(self.occupancy_map.costmap.shape[0]):
            for j in range(self.occupancy_map.costmap.shape[1]):
                if self.occupancy_map.costmap[i,j] > 0.9: 
                    self.obstacles.append({'position': np.array([i, j]), 'radius': 2.0})

        for ob in self.obstacles:
            for m in range(self.M):
                # hard constraint (doesn't work)
                self.opti.subject_to(sum([(self.x[m][j,:] - ob['position'][i])**2 for i, j in enumerate(self.dynamics.physical_state_idx)]) >= ob['radius']**2)
                # soft constraint
                # for i, j in enumerate(self.dynamics.physical_state_idx):
                #     dist = (self.x[m][j,:] - ob['position'][i])**2
                #     # collision_cost += casadi.sum2(casadi.if_else(dist >= ob['radius']**2, 0.0, ob['radius']**2 - dist)) # doesn't work
                #     # compute element wise inverse
                #     # collision_cost += casadi.sum2(1/(dist+0.1)) # this doesn't works
                #     # compute element wise inverse squared
                #     collision_cost += casadi.sum2(1/(dist+1.0)**2) # this works # 0.01 is too small

        return collision_cost

    def _general_opt_setup(self):
        ''' Initiate Casadi optimzation '''
        self.opti = Opti()
        self.x = [self.opti.variable(self.dynamics.x_shape, self.N+1) for m in range(self.M)]
        self.u = [self.opti.variable(self.dynamics.u_shape, self.N) for m in range(self.M)]
    
    def _add_dynamic_constraints(self):
        ''' Constrain states at n and n+1 to follow dynamics '''
        for k in range(self.N):
            for m in range(self.M):
                self.opti.subject_to(self.x[m][:,k+1] == \
                                     self.dynamics.propagate(self.x[m][:,k], self.u[m][:,k], self.dt))

    def _add_state_constraint(self, k, x):
        ''' Used to constrain initial and final state conditions '''

        terminal = k == -1 # if terminal state

        if not terminal:
            for i in range(self.dynamics.x_shape):
                for m in range(self.M):
                    self.opti.subject_to(self.x[m][i,k] == x[m,i])

        else:
            # only constrain position (first two states)
            for i in range(2):
                for m in range(self.M):
                    self.opti.subject_to(self.x[m][i,k] == x[m,i])
        
    def _add_input_constraint(self, k, u):
        ''' Used to constrain initial and final input conditions '''
        for i in range(self.dynamics.u_shape):
            for m in range(self.M):
                if not np.isnan(u.item(i)):
                    self.opti.subject_to(self.u[m][i,k] == u.item(i))

    def _add_obstacle_constraints(self):
        ''' Constraints physical state dimensions to not be within obstacle radii '''
        for ob in self.obstacles:
            # import ipdb; ipdb.set_trace()
            # print(self.dynamics.physical_state_idx)
            # [(x[j,:]) for i, j in enumerate(self.dynamics.physical_state_idx)]
            # [(ob['position'][i])**2 for i, j in enumerate(self.dynamics.physical_state_idx)]
            for m in range(self.M):
            # [(x[j,:] - ob['position'][i])**2 for i, j in enumerate(self.dynamics.physical_state_idx)]
                self.opti.subject_to(sum([(self.x[m][j,:] - ob['position'][i])**2 for i, j in enumerate(self.dynamics.physical_state_idx)]) >= ob['radius']**2)

    def _add_multi_agent_collision_constraints(self):
        for m1 in range(self.M):
            for m2 in range(self.M):
                if m1 == m2:
                    continue
                self.opti.subject_to((self.x[m1][0,:] - self.x[m2][0,:])**2+(self.x[m1][1,:] - self.x[m2][1,:])**2 >= self.min_allowable_dist) 
    