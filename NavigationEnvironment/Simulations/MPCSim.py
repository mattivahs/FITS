import numpy as np
import matplotlib.pyplot as plt
import copy
import jax.numpy as jnp
import time
from NavigationEnvironment.Baselines import MPC
import os
import pickle


# Initial state and goal state
x0 = jnp.array([0.0, 1.0, 0.0, 0.])
xg = jnp.array([6.0, 6.0, 0.0, 0.])

save_data = False

N_obs = 30
obstacles = []
np.random.seed(43)
for i in range(N_obs):
    x = np.random.uniform(1., 5.)
    y = np.random.uniform(1., 5.)
    r = np.random.uniform(0.1, 0.6)
    obstacles.append((jnp.array([x, y]), r))

def smooth_min(y):
    gamma = 10.0
    return - (1 / gamma) * np.log(np.sum(np.exp(- gamma * y)))

constraint_functions = [lambda x: smooth_min(np.array([(x[0:2] - c[0]).T @ (x[0:2] - c[0]) - c[1]**2 for c in obstacles]))]

bd1 = lambda y: -y[0] + 6
bd2 = lambda y: -y[1] + 6
bd3 = lambda y: y[0] + 1
bd4 = lambda y: y[1] + 1

# boundary constraints
constraint_functions.append(bd1)
constraint_functions.append(bd2)
constraint_functions.append(bd3)
constraint_functions.append(bd4)

mpc = MPC(control_freq=20,
          horizon=30,
          q_mpc=[10., 10., 1., 1.],
          r_mpc=[1., 1.],
          warmstart=True,
          soft_constraints=False,
          constraint_functions=constraint_functions,
        )

# Simulation parameters
dt = 0.01  # Time step
T = 8. # Total time
num_steps = int(T / dt)

# Dynamics function for ODE solver with control input
planned_trajs = []
controls = []
comp_times = []
h_vals = []
h = lambda x: np.min([np.linalg.norm(x[0:2] - c[0]) - c[1] for c in obstacles])

def closed_loop_sys(x, t):
    start = time.time()
    u, xt = mpc.get_control(x)
    tc = time.time() - start
    print("control takes ", tc)
    comp_times.append(tc)
    h_vals.append(h(x))
    dx = mpc.dyn.f(x) + mpc.dyn.g(x) @ u
    controls.append(u)
    planned_trajs.append(xt.T)
    return dx

# Time vector for the integration
t = np.linspace(0, T, num_steps)
trajectory = np.zeros((num_steps, 4))
trajectory[0, :] = copy.copy(x0)

# solve ODE using Euler
xnext = x0
tnext = 0.
for i in range(1, num_steps):
    xnext += dt * closed_loop_sys(xnext, tnext)
    tnext += dt
    trajectory[i, :] = copy.copy(xnext)

results = {"trajs_data": trajectory, "controls": controls, "comp_times": comp_times, "h_vals": h_vals}
fig, ax = plt.subplots()

# Initialize plot elements
for obs, r in obstacles:
    circle = plt.Circle(obs.tolist(), r, color='r', alpha=0.5, label='Obstacle')
    ax.add_patch(circle)
traj_line, = ax.plot([], [], label='Trajectory')
goal_point, = ax.plot([], [], 'go', label='Goal')
plan_line, = ax.plot([], [], label='Planned Trajectory', color='c')

traj_line.set_data(trajectory[:, 0], trajectory[:, 1])
goal_point.set_data([xg[0]], [xg[1]])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Unicycle Model Avoiding Circular Obstacle')
ax.grid()
ax.axis('equal')
ax.set_xlim(-0.1, 6.5)
ax.set_ylim(-0.1, 6.5)

plt.show()

fig, ax = plt.subplots()

ax.plot([c[0] for c in controls])
ax.plot([c[1] for c in controls])
plt.show()

if save_data:
    path_dir = os.path.dirname('./temp-data/')
    os.makedirs(path_dir, exist_ok=True)
    with open(f'./temp-data/MPC1.pkl', 'wb') as file:
        pickle.dump(results, file)