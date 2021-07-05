import numpy as np
import random
import pybullet as pb
from pybullet_utils import bullet_client as bc
import gym
from matplotlib.ticker import LinearLocator


import os
os.sys.path.append(os.path.join(os.getcwd(), '.'))
from safe_rl.navigation_obs_pb import NavigationObsPBEnv


class NavigationObsPBEnvDisc(NavigationObsPBEnv):
    def __init__(self, task={},
                        img_H=128,
                        img_W=128,
                        useRGB=True,
                        render=True,
                        doneType='fail'):
        """
        __init__: initialization

        Args:
            task (dict, optional): task information dictionary. Defaults to {}.
            img_H (int, optional): height of the observation. Defaults to 96.
            img_W (int, optional): width of the observation.. Defaults to 96.
            render (bool, optional): use pb.GUI if True. Defaults to True.
        """
        super(NavigationObsPBEnvDisc, self).__init__(   task=task,
                                                        img_H=img_H,
                                                        img_W=img_W,
                                                        useRGB=useRGB,
                                                        render=render,
                                                        doneType=doneType)

        # Discrete action space
        self.action_space = gym.spaces.Discrete(3)
        self.action_num = 3
        self.discrete_controls = np.array([ -self.action_lim[0],
                                            0.,
                                            self.action_lim[0]])

        # Fix seed
        self.seed(0)


    def getTurningRate(self, action):
        # Determine turning rate
        if not np.isscalar(action):
            action = action[0]
        w = self.discrete_controls[action]
        return w


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Test single environment in GUI
    render=False
    useRGB=True
    env = NavigationObsPBEnvDisc(render=render, useRGB=useRGB, doneType='end')
    print(env._renders)
    print("\n== Environment Information ==")
    print("- state dim: {:d}, action dim: {:d}".format(env.state_dim, env.action_dim))
    print("- state bound: {:.2f}, done type: {}".format(env.state_bound, env.doneType))
    print("- action space:", env.action_space)

    # Run 3 trials
    for i in range(2):
        print('\n== {} =='.format(i))
        if i == 0:
            obs = env.reset(random_init=False)
        else:
            obs = env.reset(random_init=True)
        for t in range(100):
            # Apply random action
            action = env.action_space.sample()
            obs, r, done, info = env.step(action)
            state = info['state']

            # Debug
            x, y, yaw = state
            l_x = info['l_x']
            g_x = info['g_x']
            print('[{}] x: {:.3f}, y: {:.3f}, l_x: {:.3f}, g_x: {:.3f}, d: {}'.format(
                t, x, y, l_x, g_x, done))
            if useRGB:
                plt.imshow(obs[:3].transpose(1,2,0))
            else:
                plt.imshow(obs[-1], cmap='Greys')
            plt.show(block=False)    # Default is a blocking call
            plt.pause(.5)
            plt.close()
            if done:
                break