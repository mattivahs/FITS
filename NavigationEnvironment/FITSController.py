from NavigationEnvironment.Solver import *
from cvxopt import matrix, solvers
import numpy as np


class FITS:
    '''MPC with full nonlinear model.'''

    def __init__(
            self,
            control_freq: int = 100,
            horizon: int = 5,
            trajectory_discretization: int = 30,
            alpha_1: float = 5,
            alpha_2: float = 10,
            warmstart: bool = True,
            use_min_formulation: bool = False,
            constraint_functions: list = None,
            ):
        '''Creates task and controller.

        Args:
            horizon (int): mpc planning horizon.
            warmstart (bool): if to initialize from previous iteration.
        '''

        self.warmstart = warmstart
        self.use_min = use_min_formulation

        # self.dyn = DynamicUnicycleModel()
        self.dyn = DIModel()

        self.state = jnp.concatenate((jnp.zeros(self.dyn.nx), 0.01*jnp.ones((horizon - 1) * self.dyn.nu)))

        self.dt = 1. / control_freq
        self.N = horizon
        self.T = (self.N) * self.dt
        self.M = trajectory_discretization


        # actuation constraints
        self.umin = self.dyn.u_min
        self.umax = self.dyn.u_max

        self.constraint_functions = constraint_functions

        # self.h_x = constraint_fun
        c_funs = [lambda x, c=c: self.h_s(x, c) for c in self.constraint_functions]

        ode_step = self.T / float(self.M)
        self.solver = DifferentiableEuler(self.dyn, self.T, ode_step, self.T / self.N, c_funs, self.J_s, dynamic_J=False)
        self.alp1 = alpha_1
        self.alp2 = alpha_2

        self.actuation_constraints = jax.jit(self.input_constraints)
        self.fs_jit = jax.jit(self.fs)
        self.gs_jit = jax.jit(self.gs)

        # min formulation
        self.softmin = jax.jit(self.smooth_min)
        self.dsoftminds = jax.jit(self.smooth_min_gradient)
        self.min_formulation = jax.jit(self.min_formulation_)

        # Compile functions
        print("### Just-in-time compilation starting ###")
        s = jnp.ones(self.dyn.nx + (self.N - 1) * self.dyn.nu)
        self.solver.integrate(s)
        self.solver.odeint(s)
        for dhds in self.solver.dhdss:
            dhds(s)
        self.solver.dJds(s)
        self.actuation_constraints(jnp.empty((0, (self.N - 1) * self.dyn.nu)), jnp.empty((0, 1)), s)
        self.fs_jit(s)
        self.gs_jit(s)
        if self.use_min and self.constraint_functions:
            self.softmin(jnp.ones(2*((self.N - 1) * self.dyn.nu) + len(self.constraint_functions)))
            self.dsoftminds(jnp.ones(2*((self.N - 1) * self.dyn.nu) + len(self.constraint_functions)),
                            jnp.ones((2*((self.N - 1) * self.dyn.nu) + len(self.constraint_functions),
                                      (self.N - 1) * self.dyn.nu)))
            self.min_formulation(s)
        print("### DONE ###")

        self.last_sol = None

    def fs(self, x):
        """
        :param x: concatenated state of initial conditions and input trajectory
        :return: vectorfield f in input trajectory space
        """
        return jnp.concatenate((self.dyn.f(x[:self.dyn.nx]) + self.dyn.g(x[:self.dyn.nx]) @ x[self.dyn.nx:(self.dyn.nx + self.dyn.nu)],
                               jnp.zeros(((self.N - 1) * self.dyn.nu))))

    def gs(self, x):
        return jnp.vstack((jnp.zeros((self.dyn.nx, (self.N - 1) * self.dyn.nu)),
                          jnp.eye((self.N - 1) * self.dyn.nu)))

    def h_s(self, x_sol, c):
        return jnp.min(c(x_sol[2:, :]))
        # return self.softmin(-c(x_sol.T))

    def init_solution(self, x0, dt, ref):
        # Optimize initial solution
        for i in range(100):
            self.get_control(x0, dt, ref)

    def J_s(self, x_sol):
        J = 10*(jnp.sum(jnp.linalg.norm(jnp.array([1., 1.]) * (x_sol[..., :2] - jnp.array([6., 6.])), axis=1))) + 0*x_sol[-1, 2:] @ x_sol[-1, 2:].T
        return J

    def J_filter(self, state):
        u_ref = jnp.clip(jnp.diag(jnp.array([-1, -1])) @ (state[:2] - jnp.array([6., 6.])), self.dyn.u_min, self.dyn.u_max)
        return jnp.linalg.norm(state[self.dyn.nx:(self.dyn.nx + self.dyn.nu)] - u_ref)

    def input_constraints(self, G_ineq, h_ineq, state):
        G_ineq = jnp.vstack([G_ineq, jnp.eye((self.N - 1) * self.dyn.nu), - jnp.eye((self.N - 1) * self.dyn.nu)])
        h_ineq = jnp.vstack([h_ineq,
                            (- self.alp2 * (state[self.dyn.nx:] - jnp.tile(self.umin, self.N - 1))).reshape(-1, 1),
                            (- self.alp2 * (jnp.tile(self.umax, self.N - 1) - state[self.dyn.nx:])).reshape(-1, 1)])
        return G_ineq, h_ineq

    def smooth_min(self, y):
        gamma = 10.0
        return - (1 / gamma) * jnp.log(jnp.sum(jnp.exp(- gamma * y)))

    def smooth_min_gradient(self, y, dhdss):
        gamma = 10.0
        exp_term = jnp.exp(-gamma * y)
        sum_exp = jnp.sum(exp_term)
        gradient = exp_term / sum_exp
        return gradient @ dhdss

    def min_formulation_(self, state):
        h_i, dhds_i = self.solver.dhdss[0](state)

        h_collection = jnp.array(h_i)
        dhds_collection = jnp.array(dhds_i)

        for i in range(1, len(self.constraint_functions)):
            h_i, dhds_i = self.solver.dhdss[i](state)
            h_collection = jnp.hstack([h_collection, h_i])
            dhds_collection = jnp.vstack([dhds_collection, dhds_i])

        dhds_collection = jnp.array(dhds_collection)
        dhds_collection = jnp.vstack([dhds_collection, 1.0*jnp.hstack([jnp.zeros(((self.N - 1) * self.dyn.nu, self.dyn.nx)), jnp.eye((self.N - 1) * self.dyn.nu)]),
                            - 1.0*jnp.hstack([jnp.zeros(((self.N - 1) * self.dyn.nu, self.dyn.nx)), jnp.eye((self.N - 1) * self.dyn.nu)])])

        h_collection = jnp.array(h_collection)
        h_collection = jnp.hstack([h_collection,
                         1.0*(state[self.dyn.nx:] - jnp.tile(self.umin, self.N - 1)),
                         1.0*(jnp.tile(self.umax, self.N - 1) - state[self.dyn.nx:])])

        h = self.softmin(0.001*h_collection)
        dhds = self.dsoftminds(h_collection, dhds_collection)

        return h, dhds

    def get_control(self, x, dt_=0.01, reference=None):
        self.state = self.state.at[:self.dyn.nx].set(jnp.array(x))

        G_ineq = jnp.empty((0, (self.N - 1) * self.dyn.nu))
        h_ineq = jnp.empty((0, 1))

        # vector fields of control affine dynamics in s
        f_s = self.fs_jit(self.state)
        g_s = self.gs_jit(self.state)

        if self.use_min:
            if self.constraint_functions:
                h, dhds = self.min_formulation(self.state)
                print(h)
                Lfh = dhds @ f_s
                Lgh = dhds @ g_s
                G_ineq = jnp.vstack([G_ineq, Lgh])
                h_ineq = jnp.vstack([h_ineq, (- 5. * h - Lfh).reshape(-1, 1)])
        else:
            for i in range(len(self.constraint_functions)):
                h, dhds = self.solver.dhdss[i](self.state)

                Lfh = dhds @ f_s
                Lgh = dhds @ g_s
                G_ineq = jnp.vstack([G_ineq, Lgh])
                h_ineq = jnp.vstack([h_ineq, (- self.alp1 * h - Lfh).reshape(-1, 1)])

            G_ineq, h_ineq = self.actuation_constraints(G_ineq, h_ineq, self.state)

        G_ineq = matrix(- np.array(G_ineq).astype(np.double))
        h_ineq = matrix(- np.array(h_ineq).astype(np.double).flatten())

        Q = matrix((1 / self.N) * 500 * np.diag(np.ones((self.N - 1) * self.dyn.nu)))

        dJds = self.solver.dJds(self.state)
        p = matrix(np.array(dJds @ g_s).astype(np.double))

        solvers.options['show_progress'] = False
        sol = solvers.qp(Q, p, G_ineq, h_ineq)

        # update control trajectory
        v = np.array(sol['x'])[:, 0]
        print(v[0])
        u_out = copy.copy(self.state[self.dyn.nx:self.dyn.nx + self.dyn.nu])
        x_out = self.solver.odeint(self.state)
        self.state = self.state.at[self.dyn.nx:].set(self.state[self.dyn.nx:] + v * dt_)

        return u_out, x_out