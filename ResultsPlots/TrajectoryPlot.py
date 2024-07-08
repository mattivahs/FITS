import matplotlib.pyplot as plt
import pickle
import numpy as np
import matplotlib.patches as patches

with open('Data/CBF.pkl', 'rb') as file:
    # Load the content
    cbf = pickle.load(file)

with open('Data/FITS.pkl', 'rb') as file:
    # Load the content
    fits = pickle.load(file)

with open('Data/MPC.pkl', 'rb') as file:
    # Load the content
    MPC = pickle.load(file)

CBF_states = cbf['trajs_data']['obs'][0]
fits_states = fits['trajs_data']['obs'][0][0:380, :]
MPC_states = MPC['trajs_data']['obs'][0][0:190, :]

fits_ref = fits['reference']
MPC_ref = MPC['reference']
CBF_ref = cbf['reference']
# Enable LaTeX rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Plot xz trajectory
fig2 = plt.figure(figsize=(5, 3.8))
ax = fig2.add_subplot(111)
ax.plot(np.array(CBF_states).T[0, :], np.array(CBF_states).T[2, :], label='CBF', color='darkorange')
ax.plot(np.array(fits_states).T[0, :], np.array(fits_states).T[2, :], label='FITS', color=(0, 100/255, 222/255))
ax.plot(np.array(MPC_states).T[0, :], np.array(MPC_states).T[2, :], label='NMPC', color=(0, 140/255, 0/255), linestyle='dashed')
rect = patches.Rectangle((-0.3, 0.6), 0.6, 0.8, linewidth=2, edgecolor=(220/255, 33/255, 77/255), facecolor='none')
# ax.plot(env.X_GOAL[:, 0], env.X_GOAL[:, 2], linestyle='dotted', color='black')

ax.add_patch(rect)

# Create a circle with a center at (0.5, 0.5) and radius 0.4
circle = patches.Circle((0., 1.), 0.5, edgecolor='black', facecolor='none', linestyle='dotted')

# Add the circle to the axes
ax.add_patch(circle)
# Set the aspect ratio of the plot to be equal
ax.set_aspect('equal', adjustable='box')
# Set the limits of the plot
ax.set_xlim(-0.55, 0.55)
ax.set_ylim(0.45, 1.55)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=3)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

# Set labels with LaTeX formatting
ax.set_xlabel(r'$p_x$\textrm{ in [m]}', fontsize=18)
ax.set_ylabel(r'$p_z$\textrm{ in [m]}', fontsize=18)

plt.savefig('TrajectoriesQuadrotor.pdf')

def max_error(traj):
    h = lambda x: np.min([x[0] + 0.3, 0.3 - x[0], x[2] - 0.6, 1.4 - x[2]])
    return np.min([h(traj[i, :]) for i in range(traj.shape[0])])


## RMSE
print("RMSE OACIS: " + str(np.mean([np.linalg.norm(fits_states[i, [0, 2]] -
       fits_ref[i,[0, 2]]) for i in range(fits_states.shape[0])])) + str(" pm ")
        + str(np.std([np.linalg.norm(fits_states[i, [0, 2]] - fits_ref[i,[0, 2]]) for i in range(fits_states.shape[0])])))

print("RMSE MPC: " + str(np.mean([np.linalg.norm(MPC_states[i, [0, 2]] -
       MPC_ref[i,[0, 2]]) for i in range(MPC_states.shape[0])])) + str(" pm ")
        + str(np.std([np.linalg.norm(MPC_states[i, [0, 2]] - MPC_ref[i,[0, 2]]) for i in range(MPC_states.shape[0])])))

print("RMSE CBF: " + str(np.mean([np.linalg.norm(CBF_states[i, [0, 2]] -
       CBF_ref[i,[0, 2]]) for i in range(CBF_states.shape[0])])) + str(" pm ")
        + str(np.std([np.linalg.norm(CBF_states[i, [0, 2]] - CBF_ref[i,[0, 2]]) for i in range(CBF_states.shape[0])])))

## CONTROL EFFORT
print("CONTROL OACIS: " + str(np.mean([np.linalg.norm(fits['trajs_data']['action'][0][i, :]) for i in range(fits_states.shape[0])]))
      + str(" pm ") + str(np.std([np.linalg.norm(fits['trajs_data']['action'][0][i, :]) for i in range(fits_states.shape[0])])))

print("CONTROL MPC: " + str(np.mean([np.linalg.norm(MPC['trajs_data']['action'][0][i, :]) for i in range(MPC_states.shape[0])]))
      + str(" pm ") + str(np.std([np.linalg.norm(MPC['trajs_data']['action'][0][i, :]) for i in range(MPC_states.shape[0])])))

print("CONTROL CBF: " + str(np.mean([np.linalg.norm(cbf['trajs_data']['action'][0][i, :]) for i in range(cbf['trajs_data']['action'][0].shape[0])]))
      + str(" pm ") + str(np.std([np.linalg.norm(cbf['trajs_data']['action'][0][i, :]) for i in range(cbf['trajs_data']['action'][0].shape[0])])))

## INFEASIBILITY RATE

## COMPUTATION TIME
print("Time OACIS: " + str(np.mean(fits['trajs_data']['controller_data'][0]['t_wall'][0][1:]))
      + str(" pm ") + str(np.std(fits['trajs_data']['controller_data'][0]['t_wall'][0][1:])))

print("Time MPC: " + str(np.mean(MPC['trajs_data']['controller_data'][0]['t_wall'][0][1:]))
      + str(" pm ") + str(np.std(MPC['trajs_data']['controller_data'][0]['t_wall'][0][1:])))

print("Time CBF: " + str(np.mean(cbf['trajs_data']['safety_filter_data'][0]['t_wall'][0][1:]))
      + str(" pm ") + str(np.std(cbf['trajs_data']['safety_filter_data'][0]['t_wall'][0][1:])))

## MINIMUM H VALUE
print("Min h_val FITS: " + str(max_error(fits_states)))
print("Min h_val CBF: " + str(max_error(MPC_states)))
print("Min h_val CBF: " + str(max_error(CBF_states)))

plt.show()