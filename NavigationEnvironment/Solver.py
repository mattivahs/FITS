'''General utility functions.'''

import jax
import jax.numpy as jnp
from math import floor
import copy


class DifferentiableEuler:
    def __init__(self, dyn, T, dt, control_dt, h_funs=[], J_fun=None, dynamic_J=False):
        self.dyn = dyn
        self.ode_jit = jax.jit(self.ode)
        self.T = T
        self.dt = dt
        self.control_dt = control_dt

        self.odeint = jax.jit(self.integrate_fori)
        self.val_grad = jax.value_and_grad(self.odeint)
        self.diff_ode_sol = jax.jit(self.val_grad)

        self.h_jits = []
        self.h_grads = []
        self.dhdss = []
        for i, h in enumerate(h_funs):
            self.h_jits.append(jax.jit(h))

            # Capture the current value of i using a default argument
            self.h_grads.append(jax.value_and_grad(lambda x, h_jit=self.h_jits[i]: self.diff_h_fun(x, h_jit)))
            self.dhdss.append(jax.jit(self.h_grads[i]))


        self.J_jit = jax.jit(J_fun)
        if not dynamic_J:
            self.J_grad = jax.grad(self.diff_J_fun)
            # self.J_grad = jax.grad(self.J_jit)
            self.dJds = jax.jit(self.J_grad)
        else:
            self.J_grad = jax.grad(self.diff_J_fun_dynamic, argnums=0)
            self.dJds = jax.jit(self.J_grad)

        self.x_sol = None

    def diff_h_fun(self, s, h_jit):
        x_sol = self.odeint(s)
        return h_jit(x_sol)

    def diff_J_fun(self, s):
        x_sol = self.odeint(s)
        return self.J_jit(x_sol) + 1*jnp.sum(s[self.dyn.nx:] ** 2)

    def diff_J_fun_dynamic(self, s, reference):
        x_sol = self.odeint(s)
        self.xsol = x_sol
        return self.J_jit(x_sol, reference) + 10*jnp.sum(s[self.dyn.nx:] ** 2)

    def ode(self, x, u):
        return self.dyn.f(x) + self.dyn.g(x) @ u

    def integrate(self, s):
        x0 = s[:self.dyn.nx]
        u_seq = s[self.dyn.nx:]
        x_sol = [x0]
        times = [0.]
        for i in range(int(self.T / self.dt)):
            ind = min(u_seq.shape[0] - self.dyn.nu, self.dyn.nu * int(floor(times[-1] / self.control_dt)))
            x_sol.append(x_sol[-1] + self.dt * self.ode_jit(x_sol[-1], u_seq[ind:(ind + self.dyn.nu)]))
            times.append(times[-1] + self.dt)

        return jnp.array(x_sol)

    def integrate_fori(self, s):
        x_sol = jnp.zeros((int(self.T / self.dt), self.dyn.nx))
        x_sol = x_sol.at[0, :].set(s[:self.dyn.nx])
        u_seq = s[self.dyn.nx:]

        def one_step(i, x):
            ind = self.dyn.nu * (jnp.floor((i-1) * self.dt / self.control_dt)).astype(int)
            x = x.at[i].set(x[i-1, :] + self.dt *
                            self.ode_jit(x[i-1, :], jax.lax.dynamic_slice(u_seq, (ind,), (self.dyn.nu,))))

            return x

        out = jax.lax.fori_loop(1, int(self.T / self.dt), one_step, x_sol)

        return out





class DynamicUnicycleModel:
    def __init__(self):
        self.nx = 4
        self.nu = 2
        self.u_min = jnp.array([-1., - 3])
        self.u_max = jnp.array([1., 3])

    def f(self, x):
        return jnp.array([x[2] * jnp.cos(x[3]), x[2] * jnp.sin(x[3]), 0., 0.])

    def g(self, x):
        return jnp.array([[0., 0.], [0., 0.], [1., 0.], [0., 1.]])

class DIModel:
    def __init__(self):
        self.nx = 4
        self.nu = 2
        self.u_min = jnp.array([-1., - 1])
        self.u_max = jnp.array([1., 1])

    def f(self, x):
        return jnp.array([x[2], x[3], 0., 0.])

    def g(self, x):
        return jnp.array([[0., 0.], [0., 0.], [1., 0.], [0., 1.]])