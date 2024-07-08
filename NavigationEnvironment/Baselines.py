from cvxopt import matrix, solvers
from NavigationEnvironment.Solver import DIModel
import jax

class CBFQP:
    def __init__(self, constraint_functions):
        self.dyn = DIModel()
        self.Q = np.diag(np.array([1., 1.]))
        self.constraint_functions = constraint_functions

        self.h_jits = []
        self.h_grads = []
        self.dhdx = []
        self.Lfhs_ = []
        self.Lfhs = []
        self.dLfhdx_ = []
        self.dLfhdx = []
        for i, h in enumerate(self.constraint_functions):
            self.h_jits.append(jax.jit(h))

            # Capture the current value of i using a default argument
            self.h_grads.append(jax.grad(lambda x, h_jit=self.h_jits[i]: h_jit(x)))
            self.dhdx.append(jax.jit(self.h_grads[i]))
            self.Lfhs_.append(lambda x, dhdxi=self.dhdx[i]: dhdxi(x) @ self.dyn.f(x))
            self.Lfhs.append(jax.jit(self.Lfhs_[i]))
            self.dLfhdx_.append(jax.grad(self.Lfhs_[i]))
            self.dLfhdx.append(jax.jit(self.dLfhdx_[i]))

    def input_saturation(self, G_ineq, h_ineq):
        G_ineq = np.vstack([G_ineq, np.array([[1., 0.],
                                              [-1., 0.],
                                              [0., 1.],
                                              [0., -1.]])])
        h_ineq = np.vstack([h_ineq, np.array([self.dyn.u_min[0], -self.dyn.u_max[0],
                                              self.dyn.u_min[1], -self.dyn.u_max[1]]).reshape(-1, 1)])
        return G_ineq, h_ineq
    def get_control(self, x, u_ref):
        G_ineq = np.empty((0, len(u_ref)))
        h_ineq = np.empty((0, 1))

        alp1 = 20
        alp2 = 0.99 * 0.25 * alp1**2

        for i in range(len(self.constraint_functions)):
            h = self.h_jits[i](x)
            Lfh = self.Lfhs[i](x)
            dLfhdx = self.dLfhdx[i](x)
            Lf2h = dLfhdx @ self.dyn.f(x)
            LgLfh = dLfhdx @ self.dyn.g(x)
            G_ineq = np.vstack([G_ineq, LgLfh])
            h_ineq = np.vstack([h_ineq, -alp2 * h - alp1 * Lfh - Lf2h - LgLfh @ u_ref])

        # G_ineq, h_ineq = self.input_saturation(G_ineq, h_ineq)
        G_ineq = matrix(- G_ineq)
        h_ineq = matrix(- h_ineq.flatten())

        Q = matrix(self.Q)
        p = matrix(np.zeros(len(u_ref)))

        solvers.options['show_progress'] = False
        sol = solvers.qp(Q, p, G_ineq, h_ineq)

        return u_ref + np.array(sol['x'])[:, 0]

'''Model Predictive Control.'''

from copy import deepcopy

import casadi as cs
import numpy as np
import time

class MPC:
    '''MPC with full nonlinear model.'''

    def __init__(
            self,
            control_freq: int = 100,
            horizon: int = 5,
            q_mpc: list = [1],
            r_mpc: list = [1],
            warmstart: bool = True,
            soft_constraints: bool = False,
            constraint_functions: list = None,
            ):
        '''Creates task and controller.

        Args:
            horizon (int): mpc planning horizon.
            q_mpc (list): diagonals of state cost weight.
            r_mpc (list): diagonals of input/action cost weight.
            warmstart (bool): if to initialize from previous iteration.
            soft_constraints (bool): Formulate the constraints as soft constraints.
            additional_constraints (list): List of additional constraints
            '''

        self.dyn = DIModel()
        self.dt = 1 / control_freq
        self.horizon = horizon
        self.Q = np.diag(np.array(q_mpc))
        self.R = np.diag(np.array(r_mpc))

        self.soft_constraints = soft_constraints
        self.warmstart = warmstart
        self.constraint_functions = constraint_functions

        self.x_prev = None
        self.u_prev = None

        self.setup_optimizer()

    def cost_fun(self):
        x = cs.SX.sym('X', self.dyn.nx)
        xg = cs.SX.sym('XG', self.dyn.nx)
        u = cs.SX.sym('U', self.dyn.nu)

        cost = (x - xg).T @ self.Q @ (x - xg) + u.T @ self.R @ u
        return cs.Function('cost', [x, u, xg], [cost])

    def dyn_func(self):
        x = cs.SX.sym('X', self.dyn.nx)
        u = cs.SX.sym('X', self.dyn.nu)

        f = cs.vertcat(x[0] + self.dt * x[2],
                            x[1] + self.dt * x[3],
                            x[2] + self.dt * u[0],
                            x[3] + self.dt * u[1])
        # f = cs.vertcat(x[0] + self.dt * (x[2] * cs.cos(x[3])),
        #                x[1] + self.dt * (x[2] * cs.sin(x[3])),
        #                x[2] + self.dt * u[0],
        #                x[3] + self.dt * u[1])
        return cs.Function('dyn', [x, u], [f])

    def setup_optimizer(self):
        '''Sets up nonlinear optimization problem.'''
        nx, nu = self.dyn.nx, self.dyn.nu

        T = self.horizon
        # Define optimizer and variables.
        opti = cs.Opti()
        # States.
        x_var = opti.variable(nx, T + 1)
        # Inputs.
        u_var = opti.variable(nu, T)
        # Initial state.
        x_init = opti.parameter(nx, 1)
        # Reference (equilibrium point or trajectory, last step for terminal cost).
        x_ref = opti.parameter(nx, T + 1)
        # Add slack variables
        n_slack = len(self.constraint_functions) * (self.horizon + 1)
        slack = opti.variable(n_slack)
        # state_slack = opti.variable(len(self.state_constraints_sym))
        # input_slack = opti.variable(len(self.input_constraints_sym))

        # cost (cumulative)
        cost = 0
        cost_func = self.cost_fun()
        slack_counter = 0
        for i in range(T):
            # Can ignore the first state cost since fist x_var == x_init.
            cost += cost_func(x_var[:, i],
                                    u_var[:, i],
                                    x_ref[:, i])
        # Terminal cost.
        cost += cost_func(x_var[:, -1],
                                u_var[:, -1],
                                x_ref[:, -1])

        for i in range(n_slack):
            cost += 100000 * slack[i]**2
        # Constraints
        dyn_func = self.dyn_func()
        for i in range(self.horizon):
            # Dynamics constraints.
            next_state = dyn_func(x_var[:, i], u_var[:, i])
            opti.subject_to(x_var[:, i + 1] == next_state)

            # actuator saturation
            opti.subject_to(u_var[:, i] <= self.dyn.u_max)
            opti.subject_to(u_var[:, i] >= self.dyn.u_min)

            # obstacle avoidance constraints
            for j, c in enumerate(self.constraint_functions):
                opti.subject_to(c(x_var[:, i]) >= slack[slack_counter])
                slack_counter += 1

        for j, c in enumerate(self.constraint_functions):
            opti.subject_to(c(x_var[:, -1]) >= slack[slack_counter])
            slack_counter += 1

        # initial condition constraints
        opti.subject_to(x_var[:, 0] == x_init)

        opti.minimize(cost)
        # Create solver (IPOPT solver in this version)
        opts = {'ipopt.print_level': 0}
        # opts = {'expand': True}
        opti.solver('ipopt', opts)
        # opti.solver('sleqp')
        self.opti_dict = {
            'opti': opti,
            'x_var': x_var,
            'u_var': u_var,
            'x_init': x_init,
            'x_ref': x_ref,
            'cost': cost
        }

    def get_control(self,
                      x
                      ):
        '''Solves nonlinear mpc problem to get next action.

        Args:
            obs (ndarray): Current state/observation.
            info (dict): Current info

        Returns:
            action (ndarray): Input/action to the task/env.
        '''

        opti_dict = self.opti_dict
        opti = opti_dict['opti']
        x_var = opti_dict['x_var']
        u_var = opti_dict['u_var']
        x_init = opti_dict['x_init']
        x_ref = opti_dict['x_ref']

        # Assign the initial state.
        opti.set_value(x_init, x)
        # Assign reference trajectory within horizon.
        goal_states = np.tile(np.array([6., 6., 0., 0.]).reshape(-1, 1), (1, self.horizon + 1))
        opti.set_value(x_ref, goal_states)

        start = time.time()
        if self.warmstart and self.x_prev is not None and self.u_prev is not None:
            # shift previous solutions by 1 step
            x_guess = deepcopy(self.x_prev)
            u_guess = deepcopy(self.u_prev)
            x_guess[:, :-1] = x_guess[:, 1:]
            u_guess[:-1] = u_guess[1:]
            opti.set_initial(x_var, x_guess)
            opti.set_initial(u_var, u_guess)
        # Solve the optimization problem.
        sol = opti.solve()

        x_val, u_val = sol.value(x_var), sol.value(u_var)
        print("control time: ", time.time() - start)
        self.x_prev = x_val
        self.u_prev = u_val

        if u_val.ndim > 1:
            action = u_val[:, 0]
        else:
            action = np.array([u_val[0]])

        return action, np.array(x_val)

