from typing import Tuple
import os
import pickle
import matplotlib as mpl
import numpy as np
import glob
import torch
from types import SimpleNamespace
import warnings


class Struct:

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def sample_with_range(
    range: Tuple[float, float], num: int, rng: np.random.Generator
):
    if range[0] == range[1]:
        return np.ones(shape=(num,)) * range[0]
    return rng.uniform(range[0], range[1], num)


def save_obj(obj, filename):
    if not filename.endswith('.pkl'):
        filename += '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(filename):
    if not filename.endswith('.pkl'):
        filename += '.pkl'
    with open(filename, 'rb') as f:
        return pickle.load(f)


def value_wrapper(agent, obs, append):
    obs_tensor = torch.FloatTensor(obs).to(agent.device)
    if obs_tensor.ndim == 3:
        obs_tensor = obs_tensor.unsqueeze(0)
    u = agent.actor(obs_tensor, append=append).detach()
    v = agent.critic(obs_tensor, u, append=append)[0]
    v = v.detach().cpu().numpy()
    return v


def load_env(config_dict):
    from env.vanilla_car_env import VanillaCarEnv

    cfg_env = config_dict['environment']
    env = VanillaCarEnv(
        max_step_train=cfg_env.max_step_train,
        max_step_eval=cfg_env.max_step_eval,
        use_append=cfg_env.use_append,
        obs_buffer=cfg_env.obs_buffer,
        g_x_fail=cfg_env.g_x_fail,
        render=False,
        img_h=cfg_env.img_h,
        img_w=cfg_env.img_w,
        fixed_init=cfg_env.fixed_init,
        sparse_reward=cfg_env.sparse_reward,
        num_traj_per_visual_init=1,
        done_type='fail',
        terminal_type=cfg_env.terminal_type,
        reward_type=cfg_env.reward_type,
        reward_goal=cfg_env.reward_goal,
        reward_wander=cfg_env.reward_wander,
        reward_obs=cfg_env.reward_obs,
    )
    env.reset()

    return env


def load_agent(config_dict):
    from agent.deployment_agent import DeploymentAgent

    cfg_env = config_dict['environment']
    config_eval = config_dict['evaluation']
    config_arch_performance = config_dict['arch_performance']
    config_arch_backup = config_dict['arch_backup']

    if config_eval.force_cpu:
        config_eval.device = 'cpu'

    if config_eval.agent != "Naive":
        config_arch_backup = config_dict['arch_backup']

    agent = SimpleNamespace(
        performance=DeploymentAgent(
            config_eval, cfg_env, config_arch_performance
        )
    )
    if config_eval.agent != "Naive":
        print("Construct backup agent")
        agent.backup = DeploymentAgent(
            config_eval, cfg_env, config_arch_backup
        )

    return agent


def plot_shield(
    ax, traj, shield_inst, c="k", lw=1.5, c_sh='g', scatter=False, s=6,
    marker="o", skip=0, alpha=1.
):
    if scatter:
        shield_inst = np.append(shield_inst, False)
        unshield_inst = np.logical_not(shield_inst)
        traj_sh = traj[shield_inst]
        traj_un = traj[unshield_inst]
        idx_sh = np.arange(start=0, stop=len(traj_sh), step=skip + 1)
        idx_un = np.arange(start=0, stop=len(traj_un), step=skip + 1)

        line_1 = ax.scatter(
            traj_sh[idx_sh, 0], traj_sh[idx_sh, 1], color=c_sh, s=s,
            marker=marker, alpha=alpha
        )
        line_0 = ax.scatter(
            traj_un[idx_un, 0], traj_un[idx_un, 1], color=c, s=s,
            marker=marker, alpha=alpha
        )

    else:
        t = 0
        for inst in np.where(shield_inst)[0]:
            line_0, = ax.plot(
                traj[t:inst + 1, 0], traj[t:inst + 1, 1], c=c, linestyle=':',
                linewidth=lw
            )
            line_1, = ax.plot(
                traj[inst:inst + 2, 0], traj[inst:inst + 2, 1], c=c_sh,
                linewidth=lw
            )
            t = inst + 1
        ax.plot(traj[t:-1, 0], traj[t:-1, 1], c=c, linestyle=':', linewidth=lw)
    return line_0, line_1


# Get experiment stats
def get_sim_results(
    model_folder, success_track_index=2, violation_track_index=-1
):
    train_details_path = os.path.join(model_folder, "train_details")
    train_dict = torch.load(train_details_path, map_location="cpu")
    train_progress = train_dict["train_progress"]

    if success_track_index is None:
        train_progress = np.array(train_progress)
        data = train_progress[:, 0]
    else:
        for i, tp in enumerate(train_progress):
            train_progress[i] = np.array(tp)
        data = train_progress[success_track_index][:, 0]

    violation_record = np.array(train_dict["violation_record"])
    episode_record = np.array(train_dict["episode_record"])

    failure_ratio = (
        violation_record[violation_track_index]
        / episode_record[violation_track_index]
    )

    return failure_ratio, data


def get_lab_results(
    model_folder, success_track_index=2, violation_track_index=-1
):
    train_details_path = os.path.join(model_folder, "train_details")
    train_dict = torch.load(train_details_path, map_location="cpu")
    train_progress = train_dict["train_progress"]

    if success_track_index is None:
        train_progress = np.array(train_progress)
        data = train_progress[:, 0]
    else:
        for i, tp in enumerate(train_progress):
            train_progress[i] = np.array(tp)
        data = train_progress[success_track_index][:, 0]

    violation_record = np.array(train_dict["violation_record"])
    episode_record = np.array(train_dict["episode_record"])

    failure_ratio = (
        violation_record[violation_track_index]
        / episode_record[violation_track_index]
    )

    return failure_ratio, data


def get_test_results(model_folder, get_shield=False, get_backup=False):
    """Gets testing results.

    Args:
        model_folder (str): the path to the model folder, under which there is
            a data folder saving test results.
        get_shield (bool, optional): the model has shielding results. Defaults
            to False.
        get_backup (bool, optional): the model has backup results. Defaults to
            False.

    Returns:
        np.ndarray: rollout results of performance policy only, of the shape
            (#tests, 3). To be more specific, each row consists of the ratio of
            successful rollout, failed rollout and the unfinished rollout.
        np.ndarray or None: rollout results of performance policy with
            shielding, of the shape (#tests, 3). To be more specific, each row
            consists of the ratio of successful rollout, failed rollout and the
            unfinished rollout. If the model does not have shielding, returns
            None.
        np.ndarray or None: rollout results of backup policy, of the shape
            (#tests, 2). To be more specific, each row consists of the ratio of
            successful rollout and failed rollout. If the model does not have
            backup, returns None.
    """
    test_stats_path = os.path.join(model_folder, "data", "test.pkl")
    with open(test_stats_path, 'rb') as f:
        test_dict = pickle.load(f)
    results_no_shield = np.mean(test_dict["results_no_shield"], axis=0)

    results_shield = None
    if get_shield:
        results_shield = np.mean(test_dict["results_shield"], axis=0)
    results_backup = None
    if get_backup:
        results_backup = np.mean(test_dict["results_backup"], axis=0)
    return results_no_shield, results_shield, results_backup


def get_final_stats(
    lab_folder, real_folder, lab_exp_name, num_tests=5, get_shield=True,
    get_backup=True, success_track_index=None, real_exp_name=None,
    violation_track_index=-1, get_all_result=True, **kwargs
):
    cnt_violation = np.empty(shape=(num_tests,))

    if get_all_result:
        results_perf_dim = (num_tests, 3)
        results_backup_dim = (num_tests, 2)
    else:
        results_perf_dim = (num_tests,)
        results_backup_dim = (num_tests,)

    succ_perf = np.empty(shape=results_perf_dim)
    succ_shield = None
    if get_shield:
        succ_shield = np.empty(shape=results_perf_dim)
    succ_backup = None
    if get_backup:
        succ_backup = np.empty(shape=results_backup_dim)
    print("== Get the results of {} ==".format(lab_exp_name))
    if real_exp_name is None:
        real_exp_name = lab_exp_name

    for seed in range(num_tests):
        # get lab train results
        model_folder = os.path.join(lab_folder, lab_exp_name + '_' + str(seed))
        cnt_violation[seed] = get_lab_results(
            model_folder, success_track_index=success_track_index,
            violation_track_index=violation_track_index
        )[0]

        # get test results
        model_folder = os.path.join(
            real_folder, real_exp_name + '_' + str(seed)
        )
        results_no_shield, results_shield, results_backup = get_test_results(
            model_folder, get_shield=get_shield, get_backup=get_backup
        )

        if get_all_result:
            succ_perf[seed] = results_no_shield
            if get_shield:
                succ_shield[seed] = results_shield
            if get_backup:
                succ_backup[seed] = results_backup
        else:
            succ_perf[seed] = results_no_shield
            if get_shield:
                succ_shield[seed] = results_shield
            if get_backup:
                succ_backup[seed] = results_backup
    return cnt_violation, succ_perf, succ_shield, succ_backup


# plot stats
def plot_mean_min_max(ax, x, y, c, lw, alpha, label):
    y_mean = np.mean(y, axis=0)
    y_min = np.min(y, axis=0)
    y_max = np.max(y, axis=0)

    ax.plot(x, y_mean, '-', color=c, label=label, linewidth=lw, markersize=3)
    ax.fill_between(x, y_min, y_max, color=c, alpha=alpha)


def plot_bar_mean_min_max(
    ax, x, y, c, ec='k', lw=1.5, width=1, label=None, capsize=3, **kwargs
):
    y_mean = np.mean(y, axis=0)
    y_min = np.min(y, axis=0)
    y_max = np.max(y, axis=0)

    yerr = np.empty(shape=(2, y_mean.shape[0]))
    yerr[0] = np.subtract(y_mean, y_min)
    yerr[1] = np.subtract(y_max, y_mean)

    if label is not None:
        bar = ax.bar(
            x, y_mean, yerr=yerr, width=width, color=c, ecolor=ec,
            linewidth=lw, label=label, capsize=capsize, **kwargs
        )
    else:
        bar = ax.bar(
            x, y_mean, yerr=yerr, width=width, color=c, ecolor=ec,
            linewidth=lw, capsize=capsize, **kwargs
        )

    # ax.fill_between(x, y_min, y_max, color=c, alpha=alpha)
    return bar


def plot_bar_all(
    ax, x, y, c, lw=1., width=1, use_hatch=True, hatch_list=[None, '/////'],
    hatch_width=1., **kwargs
):
    mpl.rcParams['hatch.linewidth'] = hatch_width
    y_means = np.mean(y, axis=0)
    assert y_means.shape[0] == len(hatch_list), "not matched"

    y_means = np.insert(y_means, 0, 0)

    for i in range(len(y_means) - 1):
        if not use_hatch:
            hatch = None
        else:
            hatch = hatch_list[i]
        bottom = np.sum(y_means[:i + 1])
        y = y_means[i + 1]
        print(bottom, y)
        ax.bar(
            x, y, width=width, bottom=bottom, color=c, linewidth=lw,
            edgecolor='w', hatch=hatch, **kwargs, zorder=0
        )
        ax.bar(
            x, y, width=width, bottom=bottom, color='none', linewidth=lw,
            edgecolor='k', **kwargs, zorder=1
        )
        bottom = y


def plot_line_mean_min_max(ax, x, y, c, lw=1., s=10, alpha=1, label=None):
    plot_x_line = False
    if isinstance(x, np.ndarray):
        plot_x_line = True
        x_mean = np.mean(x, axis=0)
        x_min = np.min(x, axis=0)
        x_max = np.max(x, axis=0)
    else:
        x_mean = x

    y_mean = np.mean(y, axis=0)
    y_min = np.min(y, axis=0)
    y_max = np.max(y, axis=0)

    ax.scatter(x_mean, y_mean, c=c, s=s)
    ax.plot([x_mean, x_mean], [y_min, y_max], linewidth=lw, c=c, alpha=alpha,
            label=label)
    if plot_x_line:
        ax.plot([x_min, x_max], [y_mean, y_mean], linewidth=lw, c=c,
                alpha=alpha)


def sign(a):
    return float(a > 0) - float(a < 0)


def rgba2rgb(rgba, background=(255, 255, 255)):
    """
    Convert rgba to rgb.

    Args:
        rgba (tuple):
        background (tuple):

    Returns:
        rgb (tuple):
    """
    row, col, ch = rgba.shape
    if ch == 3:
        return rgba
    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
    a = np.asarray(a, dtype='float32') / 255.0
    R, G, B = background
    rgb[:, :, 0] = r*a + (1.0-a) * R
    rgb[:, :, 1] = g*a + (1.0-a) * G
    rgb[:, :, 2] = b*a + (1.0-a) * B
    return np.asarray(rgb, dtype='uint8')


def results2success(results, end_type="TF"):
    """Transform the results into success ratio and return the results.

    Args:
        results (np.ndarray): array of results. 1: success, -1: failure, 0: not
            yet finished.
    """
    if end_type == "fail":
        failure = np.sum(results == -1) / results.shape[0]
        success = 1 - failure
        success_ratio = np.array([success, failure])
    elif end_type == "TF":
        success = np.sum(results == 1) / results.shape[0]
        failure = np.sum(results == -1) / results.shape[0]
        unfinish = np.sum(results == 0) / results.shape[0]
        success_ratio = np.array([success, failure, unfinish])
    return success_ratio


def calculate_diversity(trajectories, bounds, num_grids_x, num_grids_y):
    """
    Return the diversity of given trajectories measured by grid occupancy
        ratio.

    Args:
        trajectories (np.ndarray): the array of trajectories, which is of the
            shape (#trajectories, #steps, state dim)
        bounds (np.ndarray): bounds[0] consists of [x_min, x_max] and bounds[1]
            consists of [y_min, y_max].
        num_grix_x (int): the number of grids in x direction.
        num_grix_y (int): the number of grids in y direction.
    """
    x_spacing = (bounds[0, 1] - bounds[0, 0]) / num_grids_x
    y_spacing = (bounds[1, 1] - bounds[1, 0]) / num_grids_y

    occupancy_mtx = np.full(shape=(num_grids_x, num_grids_y), fill_value=False)
    for traj in trajectories:
        for step in traj:
            x = step[0]
            y = step[1]
            if x >= bounds[0, 1]:
                idx_x = num_grids_x - 1
            elif x < bounds[0, 0]:
                idx_x = 0
            else:
                idx_x = np.int(np.floor((x - bounds[0, 0]) / x_spacing))
            if y >= bounds[1, 1]:
                idx_y = num_grids_y - 1
            elif y < bounds[1, 0]:
                idx_y = 0
            else:
                idx_y = np.int(np.floor((y - bounds[1, 0]) / y_spacing))
            # print(x, y, idx_x, idx_y)
            occupancy_mtx[idx_x, idx_y] = True

    return np.sum(occupancy_mtx) / (num_grids_x*num_grids_y), occupancy_mtx


def scale_and_shift(x, old_range, new_range):
    ratio = (new_range[1] - new_range[0]) / (old_range[1] - old_range[0])
    x_new = (x - old_range[0]) * ratio + new_range[0]
    return x_new


# == discretizing ==
# modified from
# https://github.com/SafeRoboticsLab/safety_rl/blob/master/utils/utils.py
def state_to_index(grid_cells, state_bounds, state):
    """
    Transform the state into the index of the nearest grid.
    Args:
        grid_cells (tuple of ints): where the ith value is the number of
            grid_cells for ith dimension of state
        state_bounds (list of tuples):  where ith tuple contains the min and
            max value in that order of ith dimension
        state (np.ndarray): state to discretize
    Returns:
        state discretized into appropriate grid_cells
    """

    index = []
    for i in range(len(state)):
        lower_bound = state_bounds[i][0]
        upper_bound = state_bounds[i][1]
        if state[i] <= lower_bound:
            if state[i] - lower_bound < -0.1:
                warnings.warn("Dimension {} out of lower bound".format(i))
            index.append(0)
        elif state[i] >= upper_bound:
            if state[i] - upper_bound > 0.1:
                warnings.warn("Dimension {} out of upper bound".format(i))
            index.append(grid_cells[i] - 1)
        else:
            index.append(
                int(((state[i] - lower_bound) * grid_cells[i]) //
                    (upper_bound-lower_bound))
            )
    return tuple(index)


def index_to_state(grid_cells, state_bounds, discrete, mode='left'):
    """
    Transform the index of the grid into the center of that cell, an "inverse"
        of state_to_index
    Args:
        grid_cells (tuple of ints): where the ith value is the number of
            grid_cells for ith dimension of state
        state_bounds (list of tuples): where ith tuple contains the min and max
            value in that order of ith dimension
        discrete (tuple of ints): discrete state to approximate to nearest real
            value
    Returns:
        the real valued state at the center of the cell of the discrete states
    """
    state = np.zeros(len(discrete))
    for i in range(len(discrete)):
        scaling = (state_bounds[i][1] - state_bounds[i][0]) / grid_cells[i]
        state[i] = discrete[i] * scaling + state_bounds[i][0]
        if mode == 'center':
            state[i] += scaling / 2

    return state


# == margin ==
def calculate_margin_rect(s, x_y_w_h_theta, negative_inside=True):
    """
    _calculate_margin_rect: calculate the margin to the box.

    Args:
        s (np.ndarray): the state.
        x_y_w_h_theta (np.ndarray): box specification, (center_x, center_y,
            width, height, tilting angle), angle is in rad.
        negativeInside (bool, optional): add a negative sign to the distance
            if inside the box. Defaults to True.

    Returns:
        float: margin.
    """
    x, y, w, h, theta = x_y_w_h_theta
    delta_x = s[0] - x
    delta_y = s[1] - y
    delta_normal = np.abs(delta_x * np.sin(theta) - delta_y * np.cos(theta))
    delta_tangent = np.sqrt(delta_x**2 + delta_y**2 - delta_normal**2)
    margin = max(delta_tangent - w, delta_normal - h)

    if negative_inside:
        return margin
    else:
        return -margin


def calculate_margin_circle(s, c_r, negative_inside=True):
    """
    _calculate_margin_circle: calculate the margin to the circle.

    Args:
        s (np.ndarray): the state.
        c_r (circle specification): (center, radius).
        negativeInside (bool, optional): add a negative sign to the distance
            if inside the box. Defaults to True.

    Returns:
        float: margin.
    """
    center, radius = c_r
    dist_to_center = np.linalg.norm(s[:2] - center)
    margin = dist_to_center - radius

    if negative_inside:
        return margin
    else:
        return -margin


# == Plotting ==
def get_rect_vertex(center, width, height, theta):
    new_x, new_y, _ = rotatePoint(np.append(center, theta), -theta)
    new_left_bottom = (new_x - width, new_y - height, 0)
    new_left_top = (new_x - width, new_y + height, 0)
    new_right_bottom = (new_x + width, new_y - height, 0)
    new_right_top = (new_x + width, new_y + height, 0)

    place_holder = np.empty(shape=(4, 2))
    place_holder[0] = rotatePoint(new_left_bottom, theta)[:2]
    place_holder[1] = rotatePoint(new_left_top, theta)[:2]
    place_holder[2] = rotatePoint(new_right_top, theta)[:2]
    place_holder[3] = rotatePoint(new_right_bottom, theta)[:2]
    return place_holder


def plot_arc(
    center, r, thetaParam, ax, c='b', lw=1.5, orientation=0, zorder=0
):
    """
    plot_arc

    Args:
        center (np.ndarray): center.
        r (float): radius.
        thetaParam (np.ndarray): [thetaInit, thetaFinal].
        ax (matplotlib.axes.Axes)
        c (str, optional): color. Defaults to 'b'.
        lw (float, optional): linewidth. Defaults to 1.5.
        orientation (int, optional): counter-clockwise angle. Defaults to 0.
        zorder (int, optional): graph layers order. Defaults to 0.
    """
    x, y = center
    thetaInit, thetaFinal = thetaParam

    xtilde = x * np.cos(orientation) - y * np.sin(orientation)
    ytilde = y * np.cos(orientation) + x * np.sin(orientation)

    theta = np.linspace(thetaInit + orientation, thetaFinal + orientation, 100)
    xs = xtilde + r * np.cos(theta)
    ys = ytilde + r * np.sin(theta)

    ax.plot(xs, ys, c=c, lw=lw, zorder=zorder)


def plot_circle(
    center, r, ax, c='b', lw=1.5, ls='-', orientation=0, scatter=False,
    zorder=0
):
    """
    plot_circle

    Args:
        center (np.ndarray): center.
        r (float): radius.
        ax (matplotlib.axes.Axes)
        c (str, optional): color. Defaults to 'b'.
        lw (float, optional): linewidth. Defaults to 1.5.
        ls (str, optional): linestyle. Defaults to '-'.
        orientation (int, optional): counter-clockwise angle. Defaults to 0.
        scatter (bool, optional): show center or not. Defaults to False.
        zorder (int, optional): graph layers order. Defaults to 0.
    """
    x, y = center
    xtilde = x * np.cos(orientation) - y * np.sin(orientation)
    ytilde = y * np.cos(orientation) + x * np.sin(orientation)

    theta = np.linspace(0, 2 * np.pi, 200)
    xs = xtilde + r * np.cos(theta)
    ys = ytilde + r * np.sin(theta)
    ax.plot(xs, ys, c=c, lw=lw, linestyle=ls, zorder=zorder)
    if scatter:
        ax.scatter(xtilde + r, ytilde, c=c, s=80)
        ax.scatter(xtilde - r, ytilde, c=c, s=80)
        print(xtilde + r, ytilde, xtilde - r, ytilde)


def plot_line(point1, point2, ax, c='b', lw=1.5, ls='-', zorder=0):
    x1, y1 = point1[:2]
    x2, y2 = point2[:2]

    ax.plot([x1, x2], [y1, y2], c=c, ls=ls, lw=lw, zorder=zorder)


def plot_rect(
    center, width, height, theta, ax, c='b', lw=1.5, ls='-', zorder=0
):
    vertices = get_rect_vertex(center, width, height, theta)

    order = [[0, 1], [1, 2], [2, 3], [3, 0]]
    for (i, j) in order:
        plot_line(
            vertices[i], vertices[j], ax, c=c, ls=ls, lw=lw, zorder=zorder
        )


def rotatePoint(state, orientation):
    """
    rotatePoint

    Args:
        state (np.ndarray): (x, y) position.
        orientation (int, optional): counter-clockwise angle.

    Returns:
        np.ndarray: rotated state.
    """
    x, y, theta = state
    xtilde = x * np.cos(orientation) - y * np.sin(orientation)
    ytilde = y * np.cos(orientation) + x * np.sin(orientation)
    thetatilde = theta + orientation

    return np.array([xtilde, ytilde, thetatilde])


# == Generate Toy Dataset ==
def target_margin(state, goal_loc, goal_radius):
    s = state[:2]
    c_r = [goal_loc, goal_radius]
    target_margin = calculate_margin_circle(s, c_r, negative_inside=True)
    return target_margin


def safety_margin(state, bounds, obs_dict):
    s = state[:2]

    x, y = (bounds[:, 0] + bounds[:, 1])[:2] / 2.0
    w, h = (bounds[:, 1] - bounds[:, 0])[:2] / 2.0
    x_y_w_h_theta = [x, y, w, h, 0]
    boundary_margin = calculate_margin_rect(
        s, x_y_w_h_theta, negative_inside=True
    )
    obstacle_list = []

    for obs_info in obs_dict:
        loc = obs_info['loc']
        if obs_info['obs_type'] == 'circle':
            radius = obs_info['radius']
            obstacle_list += [
                calculate_margin_circle(
                    s, [loc, radius], negative_inside=False
                )
            ]
        else:
            width = obs_info['width']
            height = obs_info['height']
            theta = obs_info['theta']
            x_y_w_h_theta = [loc[0], loc[1], width, height, theta]
            obstacle_list += [
                calculate_margin_rect(s, x_y_w_h_theta, negative_inside=False)
            ]

    obstacle_margin = np.max(obstacle_list)
    safety_margin = max(obstacle_margin, boundary_margin)

    return safety_margin


def get_occupancy_map(bounds, task, add_goal=True):
    obs_dict = task['obs_dict']
    goal_loc = task['goal_loc']
    goal_radius = task['goal_radius']
    grid_cells = task['grid_cells']

    # channel 1: obstacle; channel 2: goal
    occ_map = np.zeros((2,) + grid_cells, dtype='uint8')

    it = np.nditer(occ_map[0], flags=['multi_index'])

    while not it.finished:
        idx = it.multi_index
        state = index_to_state(grid_cells, bounds, idx, mode='center')
        g_x = safety_margin(state, bounds, obs_dict)
        l_x = target_margin(state, goal_loc, goal_radius)

        if g_x > 0:
            occ_map[0][idx] = 255
        elif add_goal:
            if l_x <= 0:
                occ_map[1][idx] = 255

        it.iternext()

    return occ_map


def get_feasible_rect(
    center, width, height, theta, goal_loc, goal_radius, dist_thr, step_size,
    min_width
):
    tmp_width = width
    flag_infeasible = True

    while flag_infeasible:
        vertices = get_rect_vertex(center, tmp_width, height, theta)
        distances = np.linalg.norm(vertices - goal_loc, axis=1)
        if np.all(distances >= dist_thr + goal_radius):
            flag_infeasible = False
        else:
            tmp_width -= step_size
        if tmp_width < min_width:
            return None
    return tmp_width


def get_feasible_circle(
    center, radius, goal_loc, goal_radius, dist_thr, min_radius
):
    distance = np.linalg.norm(goal_loc - center)
    tmp_radius = min(distance - (goal_radius+dist_thr), radius)
    if tmp_radius < min_radius:
        return None
    else:
        return tmp_radius


def trans2yaw_range(_range):
    low, high = _range
    if low < 0:
        if np.abs(high) < 1e-8:  # precision issue.
            return np.array([[low + 2 * np.pi, 2 * np.pi]])
        else:  # yaw range is specified in positive numbers.
            return np.array([
                [low + 2 * np.pi, 2 * np.pi],
                [0, high],
            ])
    else:
        return _range[np.newaxis]


def eval_only(model):
    for _, param in model.named_parameters():
        param.requires_grad = False
    model.eval()


def check_grad_norm(model):
    total_norm = 0
    parameters = [
        p for p in model.parameters() if p.grad is not None and p.requires_grad
    ]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item()**2
    total_norm = total_norm**0.5
    return total_norm


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0-tau) + param.data * tau
        )


def save_model(model, step, logs_path, types, max_model=None):
    start = len(types) + 1
    os.makedirs(logs_path, exist_ok=True)
    if max_model is not None:
        model_list = glob.glob(os.path.join(logs_path, '*.pth'))
        if len(model_list) > max_model - 1:
            min_step = min([
                int(li.split('/')[-1][start:-4]) for li in model_list
            ])
            os.remove(
                os.path.join(logs_path, '{}-{}.pth'.format(types, min_step))
            )
    logs_path = os.path.join(logs_path, '{}-{}.pth'.format(types, step))
    torch.save(model.state_dict(), logs_path)
    print('=> Save {} after [{}] updates'.format(logs_path, step))


# == policy distribution bound ==
def get_kl_bound(reward, kl_div, N, delta):
    R = (kl_div + np.log(2 * np.sqrt(N) / delta)) / (2*N)
    reg = np.sqrt(R)
    return reward - reg


def get_renyi_bound(reward, renyi_div, N, delta):
    log_term_r = np.log(
        2 * np.sqrt(N) / ((delta / 2)**3)
    )  # use actual delta, different from Alec's code
    Rg_r = float((renyi_div+log_term_r) / N)
    reg = np.sqrt(Rg_r / 2)
    return reward - reg


# == frame stacking and skipping ==
def get_frames(prev_obs, traj_size, frame_skip):
    """
    Assume prev_obs in tensor
    """
    traj_cover = (traj_size-1) * frame_skip + traj_size
    default_frame_seq = np.arange(0, traj_cover, (frame_skip + 1))

    if len(prev_obs) == 1:
        seq = np.zeros((traj_size), dtype='int')
    elif len(prev_obs) < traj_cover:  # always pick zero (most recent one)
        seq_random = np.random.choice(
            np.arange(1, len(prev_obs)), traj_size - 1, replace=True
        )
        seq_random = np.sort(seq_random)  # ascending
        seq = np.insert(seq_random, 0, 0)  # add to front, then flipped
    else:
        seq = default_frame_seq
    seq = np.flip(seq)  # since prev_obs appends left
    obs_stack = torch.cat([prev_obs[obs_ind] for obs_ind in seq])
    return obs_stack


# == shielding ==
def check_shielding(
    backup, shield_dict, observation, action, append, context_backup=None,
    state=None, policy=None, context_policy=None
):
    """
    Checks if shielding is needed. Currently, the latent is equivalent to
    context.

    Args:
        backup (object): a backup agent consisting of actor and critic
        shield_dict (dict): a dictionary consisting of shielding-related
            hyper-parameters.
        observation (np.ndarray or torch.tensor): the observation.
        action (np.ndarray or torch.tensor): action from the policy.
        append (np.ndarray or torch.tensor): the extra information that is
            appending after conv layers.
        context_backup (np.ndarray or torch.tensor, optional): the variable
            inducing policy distribution. It can be latent directly from a
            distribution or after encoder. Defaults to None.
        state (np.ndarray or torch.tensor): the real state. Defaults to None.
        simulator (object, optional): the environment on which we rollout
            trajectories. Defaults to None.

    Returns:
        torch.tensor: flags representing whether the shielding is necessary
    """
    if isinstance(state, np.ndarray):
        state = torch.FloatTensor(state).to(backup.device)
    if isinstance(observation, np.ndarray):
        observation = torch.from_numpy(observation).to(backup.device)
    back_to_numpy = False
    if isinstance(action, np.ndarray):
        action = torch.FloatTensor(action).to(backup.device)
        back_to_numpy = True
    if isinstance(append, np.ndarray):
        append = torch.FloatTensor(append).to(backup.device)

    # make sure the leading dim is the same
    if observation.dim() == 3:
        observation = observation.unsqueeze(0)
    if state is not None:
        if state.dim() == 1:
            state = state.unsqueeze(0)
    if action.dim() == 1:
        action = action.unsqueeze(0)
    if append.dim() == 1:
        append = append.unsqueeze(0)

    leading_equal = ((observation.shape[0] == action.shape[0])
                     and (observation.shape[0] == append.shape[0]))
    if state is not None:
        leading_equal = ((state.shape[0] == action.shape[0])
                         and (state.shape[0] == append.shape[0]))

    if not leading_equal:
        print(observation.shape, append.shape, action.shape)
        raise ValueError("The leading dimension is not the same!")
    shield_type = shield_dict['type']

    if shield_type == 'value':
        if not backup.critic_has_act_ind:
            action = action[:, :-1]
        safe_value = backup.critic(
            observation, action, append=append, latent=context_backup
        )[0].data.squeeze(1)
        shield_flag = safe_value > shield_dict['threshold']
        info = {}
    elif shield_type == 'rej':
        safe_thr = shield_dict['threshold']
        max_resample = shield_dict['max_resample']
        cnt_resample = 0
        resample_flag = True
        while resample_flag:
            if cnt_resample == max_resample:  # resample budget
                break
            if not backup.critic_has_act_ind:
                action = action[:, :-1]
            safe_value = backup.critic(
                observation, action, append=append, latent=context_backup
            )[0].data.squeeze(1)
            shield_flag = (safe_value > safe_thr)
            resample_flag = torch.any(shield_flag)
            if resample_flag:
                if context_policy is not None:
                    context_policy_resample = context_policy[shield_flag]
                else:
                    context_policy_resample = None
                a_resample, _ = policy.actor.sample(
                    observation[shield_flag], append=append[shield_flag],
                    latent=context_policy_resample
                )
                if not backup.critic_has_act_ind:
                    action[shield_flag] = a_resample.data.clone()
                else:
                    action[shield_flag, :-1] = a_resample.data.clone()
                cnt_resample += 1
        if back_to_numpy:
            action_final = action.cpu().numpy()
        else:
            action_final = action.clone()
        info = {'action_final': action_final}
    return shield_flag, info
