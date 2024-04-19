'''A PID example on a quadrotor.'''

import os
import pickle
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p

from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.controllers.pid_rl.pid_rl import obs2state, state2obs
from matplotlib.ticker import FormatStrFormatter

import torch
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def run(gui=True, n_episodes=1, n_steps=None, save_data=False, curr_path='.'):
    '''The main function running PID experiments.

    Args:
        gui (bool): Whether to display the gui and plot graphs.
        n_episodes (int): The number of episodes to execute.
        n_steps (int): The total number of steps to execute.
        save_data (bool): Whether to save the collected experiment data.
    '''

    # Create the configuration dictionary.
    CONFIG_FACTORY = ConfigFactory()
    config = CONFIG_FACTORY.merge()

    config.task_config['gui'] = gui

    # Create an environment
    env_func = partial(make,
                       config.task,
                       **config.task_config
                       )

    # Create controller.
    ctrl = make(config.algo,
                env_func,
                **config.task_config,
                output_dir=curr_path + '/temp')

    obs, _ = ctrl.env.reset(seed=42)

    if config.task_config.task == 'traj_tracking' and gui is True:
        if config.task_config.quad_type == 2:
            ref_3D = np.hstack([ctrl.env.X_GOAL[:, [0]], np.zeros(ctrl.env.X_GOAL[:, [0]].shape), ctrl.env.X_GOAL[:, [2]]])
        else:
            ref_3D = ctrl.env.X_GOAL[:, [0, 2, 4]]

        for i in range(10, ctrl.env.X_GOAL.shape[0], 10):
            p.addUserDebugLine(lineFromXYZ=[ref_3D[i - 10, 0], ref_3D[i - 10, 1], ref_3D[i - 10, 2]],
                               lineToXYZ=[ref_3D[i, 0], ref_3D[i, 1], ref_3D[i, 2]],
                               lineColorRGB=[1, 0, 0],
                               physicsClientId=ctrl.env.PYB_CLIENT)

    ctrl.load(curr_path + "/LearnedModels/")
    # ctrl.learn(ctrl.env)
    # ctrl.save(curr_path + "/LearnedModels/")

    # Run the experiment.
    experiment = BaseExperiment(ctrl.env, ctrl)
    ctrl.env.render()
    trajs_data, metrics = experiment.run_evaluation(n_episodes=n_episodes, n_steps=n_steps, seeds=[48])

    predicted_states = []
    obs, info = ctrl.env.reset(seed=48)
    next_state = obs
    predicted_states.append(next_state)

    # for i in range(len(trajs_data['action'][0])):
    #     a = torch.tensor(trajs_data['action'][0][i, :], dtype=torch.float32, device=device)
    #     next_state += 0.02 * np.array(ctrl.prior_dyn(next_state, a.detach().cpu().numpy())).flatten()
    #     # next_state = torch.from_numpy(np.float32(ctrl.prior_dyn_batch(next_state.cpu().numpy().T, a.cpu().numpy().T))).to(device).T
    #     predicted_states.append(next_state.flatten())
    experiment.close()


    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(trajs_data['state'][0][:, 0], trajs_data['state'][0][:, 2])
    # ax.plot([p_[0]for p_ in predicted_states], [p_[2]for p_ in predicted_states])
    # plt.show()

    if save_data:
        results = {'trajs_data': trajs_data, 'metrics': metrics}
        path_dir = os.path.dirname('./temp-data/')
        os.makedirs(path_dir, exist_ok=True)
        with open(f'./temp-data/{config.algo}_data_{config.task_config.task}.pkl', 'wb') as file:
            pickle.dump(results, file)

    plot_tracking(trajs_data['obs'][0], trajs_data['action'][0], ctrl.env, ctrl)


def plot_tracking(state_stack, input_stack, env, ctrl):
    model = env.symbolic
    stepsize = model.dt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.array(state_stack).transpose()[0, :], np.array(state_stack).transpose()[2, :], color='b')
    ref = ctrl.reference.cpu().numpy()
    ax.plot(ref[:, 0], ref[:, 1], color='r')
    plt.show()
def post_analysis(state_stack, input_stack, env):
    '''Plots the input and states to determine iLQR's success.

    Args:
        state_stack (ndarray): The list of observations of iLQR in the latest run.
        input_stack (ndarray): The list of inputs of iLQR in the latest run.
    '''
    model = env.symbolic
    stepsize = model.dt

    plot_length = np.min([np.shape(input_stack)[0], np.shape(state_stack)[0]])
    times = np.linspace(0, stepsize * plot_length, plot_length)

    reference = env.X_GOAL
    reference = np.tile(reference.reshape(1, model.nx), (plot_length, 1))

    # Plot states
    fig, axs = plt.subplots(model.nx)
    for k in range(model.nx):
        axs[k].plot(times, np.array(state_stack).transpose()[k, 0:plot_length], label='actual')
        axs[k].plot(times, reference.transpose()[k, 0:plot_length], color='r', label='desired')
        axs[k].set(ylabel=env.STATE_LABELS[k] + f'\n[{env.STATE_UNITS[k]}]')
        axs[k].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if k != model.nx - 1:
            axs[k].set_xticks([])
    axs[0].set_title('State Trajectories')
    axs[-1].legend(ncol=3, bbox_transform=fig.transFigure, bbox_to_anchor=(1, 0), loc='lower right')
    axs[-1].set(xlabel='time (sec)')

    # Plot inputs
    _, axs = plt.subplots(model.nu)
    if model.nu == 1:
        axs = [axs]
    for k in range(model.nu):
        axs[k].plot(times, np.array(input_stack).transpose()[k, 0:plot_length])
        axs[k].set(ylabel=f'input {k}')
        axs[k].set(ylabel=env.ACTION_LABELS[k] + f'\n[{env.ACTION_UNITS[k]}]')
        axs[k].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axs[0].set_title('Input Trajectories')
    axs[-1].set(xlabel='time (sec)')

    plt.show()


if __name__ == '__main__':
    run()
