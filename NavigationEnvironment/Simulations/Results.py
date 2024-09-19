import numpy as np
import pickle
import matplotlib.pyplot as plt

N_obs = 30
obstacles = []
np.random.seed(43)
for i in range(N_obs):
    x = np.random.uniform(1., 5.)
    y = np.random.uniform(1., 5.)
    r = np.random.uniform(0.1, 0.6)
    obstacles.append((np.array([x, y]), r))

with open('temp-data/FITS.pkl', 'rb') as file:
    # Load the content
    FITS = pickle.load(file)

with open('temp-data/CBF.pkl', 'rb') as file:
    # Load the content
    CBF = pickle.load(file)

with open('temp-data/CBF1.pkl', 'rb') as file:
    # Load the content
    CBF1 = pickle.load(file)

with open('temp-data/MPC1.pkl', 'rb') as file:
    # Load the content
    MPC = pickle.load(file)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig, ax = plt.subplots(figsize=(5, 2.8))

for obs, r in obstacles:
    circle = plt.Circle(obs, r, facecolor=(220/255, 33/255, 77/255, 0.5), edgecolor=(220/255, 33/255, 77/255))
    ax.add_patch(circle)
plt.scatter(6, 6, marker='*', s=200, color=(0, 140/255, 0))

plt.plot(CBF['trajs_data'][:,0], CBF['trajs_data'][:, 1], color='darkorange', linewidth=2, label='CBF')
plt.plot(CBF1['trajs_data'][:,0], CBF1['trajs_data'][:, 1], color='darkorange', linewidth=2)
plt.plot(FITS['trajs_data'][:,0], FITS['trajs_data'][:, 1], color=(0, 100/255, 222/255), label='FITS')
plt.plot(MPC['trajs_data'][:,0], MPC['trajs_data'][:, 1], linestyle='dashed', color=(0, 140/255, 0/255), linewidth=2, label='NMPC')

# ax.set_xlim(0., 0.55)
# ax.set_ylim(0.45, 1.55)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=3)

# Set labels with LaTeX formatting
ax.set_xlabel(r'$p_x$\textrm{ in [m]}', fontsize=18)
ax.set_ylabel(r'$p_y$\textrm{ in [m]}', fontsize=18)
plt.tight_layout()
plt.subplots_adjust(bottom=0.19)

# plt.savefig('TrajectoriesClutter.pdf')
plt.show()

