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

import torch
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def run(gui=False, n_episodes=1, n_steps=None, save_data=False, curr_path='.'):
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

    custom_trajectory = False
    if config.task_config.task == 'traj_tracking' and config.task_config.task_info.trajectory_type == 'custom':
        custom_trajectory = True
        config.task_config.task_info.trajectory_type = 'circle'  # Placeholder
        config.task_config.randomized_init = False
        config.task_config.init_state = np.zeros((12, 1))

    # Create an environment
    env_func = partial(make,
                       config.task,
                       **config.task_config
                       )

    # Create controller.
    ctrl = make(config.algo,
                env_func,
                **config.algo_config,
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
    ctrl.learn(ctrl.env)
    # ctrl.save(curr_path + "/LearnedModels/")

    # Run the experiment.
    experiment = BaseExperiment(ctrl.env, ctrl)
    # ctrl.env.render()
    trajs_data, metrics = experiment.run_evaluation(n_episodes=n_episodes, n_steps=n_steps, seeds=[42])

    predicted_states = []
    obs, info = ctrl.env.reset(seed=42)
    next_state = obs2state(obs)
    predicted_states.append(next_state.detach().cpu().numpy().flatten())

    for i in range(10):
        a = torch.tensor(trajs_data['action'][0][i, :], dtype=torch.float32, device=device)
        next_state = ctrl.dynModel.forward(next_state, a)
        predicted_states.append(next_state.detach().cpu().numpy().flatten())
    experiment.close()


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(trajs_data['state'][0][:, 0], trajs_data['state'][0][:, 2])
    ax.plot([p_[0]for p_ in predicted_states], [p_[1]for p_ in predicted_states])
    plt.show()

    if save_data:
        results = {'trajs_data': trajs_data, 'metrics': metrics}
        path_dir = os.path.dirname('./temp-data/')
        os.makedirs(path_dir, exist_ok=True)
        with open(f'./temp-data/{config.algo}_data_{config.task_config.task}.pkl', 'wb') as file:
            pickle.dump(results, file)

    iterations = len(trajs_data['action'][0])
    for i in range(iterations):
        # Step the environment and print all returned information.
        obs, reward, done, info, action = trajs_data['obs'][0][i], trajs_data['reward'][0][i], trajs_data['done'][0][i], trajs_data['info'][0][i], trajs_data['action'][0][i]

        # Print the last action and the information returned at each step.
        # print(i, '-th step.')
        # print(action, '\n', obs, '\n', reward, '\n', done, '\n', info, '\n')


if __name__ == '__main__':
    run()
