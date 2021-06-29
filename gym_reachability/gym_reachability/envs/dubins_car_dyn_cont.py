# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu        ( kaichieh@princeton.edu )

import numpy as np
import gym
from .env_utils import calculate_margin_circle, calculate_margin_rect


class DubinsCarDynCont(object):

    def __init__(self, doneType='toEnd'):
        # State bounds.
        self.bounds = np.array([[-1.1, 1.1],  # axis_0 = state, axis_1 = bounds.
                                [-1.1, 1.1],
                                [0, 2*np.pi]])
        self.low = self.bounds[:, 0]
        self.high = self.bounds[:, 1]

        # Dubins car parameters.
        self.alive = True
        self.time_step = 0.05
        self.speed = 0.5 # v

        # Control parameters.
        self.R_turn = .6
        self.max_turning_rate = np.array([self.speed / self.R_turn],
            dtype=np.float32) # w
        self.action_space = gym.spaces.Box(
            -self.max_turning_rate, self.max_turning_rate)

        # Constraint set parameters.
        self.constraint_center = None
        self.constraint_radius = None
        self.cons_neg_inside = True

        # Target set parameters.
        self.target_center = None
        self.target_radius = None

        # Internal state.
        self.state = np.zeros(3)
        self.doneType = doneType

        # Set random seed.
        self.set_seed(0)

        # Cost Params
        self.targetScaling = 1.
        self.safetyScaling = 1.


    def reset(self, start=None, theta=None, sample_inside_obs=False, sample_inside_tar=True):
        """ Reset the state of the environment.

        Args:
            start: Which state to reset the environment to. If None, pick the
                state uniformly at random.

        Returns:
            The state the environment has been reset to.
        """
        if start is None:
            x_rnd, y_rnd, theta_rnd = self.sample_random_state(
                sample_inside_obs=sample_inside_obs,
                sample_inside_tar=sample_inside_tar,
                theta=theta)
            self.state = np.array([x_rnd, y_rnd, theta_rnd])
        else:
            self.state = start
        return np.copy(self.state)


    def sample_random_state(self, sample_inside_obs=False, sample_inside_tar=True, theta=None):
        # random sample `theta`
        if theta is None:
            theta_rnd = 2.0 * np.random.uniform() * np.pi
        else:
            theta_rnd = theta

        # random sample [`x`, `y`]
        flag = True
        while flag:
            rnd_state = np.random.uniform(low=self.low[:2], high=self.high[:2])
            l_x = self.target_margin(rnd_state)
            g_x = self.safety_margin(rnd_state)

            # if l_x == None:
            #     terminal = (g_x > 0)
            # else:
            #     terminal = (g_x > 0) or (l_x <= 0)
            # flag = terminal and keepOutOf

            if (not sample_inside_obs) and (g_x > 0):
                flag = True
            elif (not sample_inside_tar) and (l_x <= 0):
                flag = True
            else:
                flag = False
        x_rnd, y_rnd = rnd_state

        return x_rnd, y_rnd, theta_rnd


#== Dynamics ==
    def step(self, action):
        """ Evolve the environment one step forward under given input action.

        Args:
            action: Input action.

        Returns:
            Tuple of (next state, signed distance of current state, whether the
            episode is done, info dictionary).
        """
        x, y, theta = self.state.copy()

        l_x_cur = self.target_margin(self.state[:2])
        g_x_cur = self.safety_margin(self.state[:2])

        state = self.integrate_forward(self.state, action)
        self.state = state

        return np.copy(self.state)

        # # done
        # if self.doneType == 'toEnd':
        #     done = not self.check_within_bounds(self.state)
        # else:
        #     assert self.doneType == 'TF', 'invalid doneType'
        #     fail = g_x_cur > 0
        #     success = l_x_cur <= 0
        #     done = fail or success

        # if done:
        #     self.alive = False

        # return np.copy(self.state), done


    def integrate_forward(self, state, u):
        """ Integrate the dynamics forward by one step.

        Args:
            x: Position in x-axis.
            y: Position in y-axis
            theta: Heading.
            u: Contol input.

        Returns:
            State variables (x,y,theta) integrated one step forward in time.
        """
        x, y, theta = state

        x = x + self.time_step * self.speed * np.cos(theta)
        y = y + self.time_step * self.speed * np.sin(theta)
        theta = np.mod(theta + self.time_step * u, 2*np.pi)
        
        state = np.array([x, y, theta])
        return state


#== Setting Hyper-Parameter Functions ==
    def set_seed(self, seed):
        """ Set the random seed.

        Args:
            seed: Random seed.
        """
        self.seed_val = seed
        self.action_space.seed(self.seed_val)


    def set_bounds(self, bounds):
        """ Set state bounds.

        Args:
            bounds: Bounds for the state.
        """
        self.bounds = bounds

        # Get lower and upper bounds
        self.low = np.array(self.bounds)[:, 0]
        self.high = np.array(self.bounds)[:, 1]


    def set_speed(self, speed=.5):
        self.speed = speed
        self.max_turning_rate = np.array([self.speed / self.R_turn],
            dtype=np.float32) # w
        self.action_space = gym.spaces.Box(
            -self.max_turning_rate, self.max_turning_rate)
        self.action_space.seed(self.seed_val)


    def set_time_step(self, time_step=.05):
        self.time_step = time_step


    def set_constraint(self, center, radius, cons_neg_inside):
        self.constraint_center = center
        self.constraint_radius = radius
        self.cons_neg_inside = cons_neg_inside


    def set_target(self, center, radius):
        self.target_center = center
        self.target_radius = radius


    def set_radius_rotation(self, R_turn=.6, verbose=False):
        self.R_turn = R_turn
        self.max_turning_rate = np.array([self.speed / self.R_turn],
            dtype=np.float32) # w
        self.action_space = gym.spaces.Box(
            -self.max_turning_rate, self.max_turning_rate)
        self.action_space.seed(self.seed_val)


    def set_radius(self, target_radius=.3, constraint_radius=1., R_turn=.6):
        self.target_radius = target_radius
        self.constraint_radius = constraint_radius
        self.set_radius_rotation(R_turn=R_turn)


#== Getting Functions ==
    def check_within_bounds(self, state):
        for dim, bound in enumerate(self.bounds):
            flagLow = state[dim] < bound[0]
            flagHigh = state[dim] > bound[1]
            if flagLow or flagHigh:
                return False
        return True


#== Compute Margin ==
    def safety_margin(self, s):
        """ Computes the margin (e.g. distance) between state and failue set.

        Args:
            s: ub-state, consisting of [x, y].

        Returns:
            Margin for the state s.
        """
        x, y = (self.low + self.high)[:2] / 2.0
        w, h = (self.high - self.low)[:2]
        boundary_margin = calculate_margin_rect(s, [x, y, w, h], negativeInside=True)
        g_xList = [boundary_margin]

        if self.constraint_center is not None and self.constraint_radius is not None:
            g_x = calculate_margin_circle(s, [self.constraint_center, self.constraint_radius],
                negativeInside=self.cons_neg_inside)
            g_xList.append(g_x)

        safety_margin = np.max(np.array(g_xList))
        return self.safetyScaling * safety_margin


    def target_margin(self, s):
        """ Computes the margin (e.g. distance) between state and target set.

        Args:
            s: sub-state, consisting of [x, y].

        Returns:
            Margin for the state s.
        """
        if self.target_center is not None and self.target_radius is not None:
            target_margin = calculate_margin_circle(s, [self.target_center, self.target_radius],
                    negativeInside=True)
            return self.targetScaling * target_margin
        else:
            return None