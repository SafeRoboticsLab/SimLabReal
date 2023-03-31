# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for training prior of PAC-Shield-Perf.

This file implements the training of the prior policy with latent-conditioned
performance policy and backup policy without conditioning on latent variable,
which is the method PAC-Shield-Perf in the paper.

The performance policy is trained with task rewards and diversity reward that
maximizes mutual information (MI) between the state and the latent
(observation-marginal MI) implemented in SAC_maxEnt.py, or maximizes MI between
the action and the latent given the state (observation-conditional MI). The
backup policy is trained with discounted safety Bellman equation.
"""

from typing import Optional, Tuple
import os
import time
import numpy as np
import torch
import wandb
import copy

from agent.base_training import BaseTraining
from agent.scheduler import StepLRMargin, StepLR
from agent.SAC_mini import SAC_mini
from agent.SAC_maxEnt import SAC_maxEnt
from agent.SAC_dads_prior import SAC_dads_prior
from utils.misc import check_shielding
from env.vec_env import VecEnvWrapper
from env.base_env import BaseEnv


class PolicyPriorPerfLatent(BaseTraining):

    def __init__(
        self, cfg, cfg_performance, cfg_backup, cfg_env, verbose: bool = True
    ):
        super(PolicyPriorPerfLatent,
              self).__init__(cfg, cfg_env, cfg_performance.train)
        #! This bool parameter controls whether using action relabeling. We
        #! argue that this parameter should always be True. We add the option
        #! here for testing the difference.
        self.action_relabel = cfg.action_relabel

        print("= Constructing performance agent")
        self.perf_diversity_type = cfg.perf_diversity_type
        if self.perf_diversity_type == 'dynamics':
            agent_class = SAC_dads_prior
        elif self.perf_diversity_type == 'state':
            agent_class = SAC_maxEnt
            self.disc_batch_size = cfg_performance.train.disc_batch_size
            self.disc_recent_size = cfg_performance.train.disc_recent_size
            self.disc_num_update_per_opt = cfg.disc_update_per_opt
        else:
            raise ValueError(
                "Not supported diversity type: {}".format(
                    self.perf_diversity_type
                )
            )

        self.performance = agent_class(
            cfg_performance.train, cfg_performance.arch, cfg_env
        )
        self.performance.build_network(verbose=verbose)

        print("= Constructing backup agent")
        self.backup = SAC_mini(cfg_backup.train, cfg_backup.arch, cfg_env)
        self.backup.build_network(verbose=verbose)
        self.get_l_x_ra = False
        if cfg_backup.train.mode == 'RA':
            self.get_l_x_ra = True

        # Boostrap probability - annealing to 1 and in the timescale of steps.
        self.eta_scheduler = StepLRMargin(
            init_value=cfg.eta, period=cfg.eta_period, threshold=0,
            decay=cfg.eta_decay, end_value=cfg.eta_end, goal_value=1
        )
        self.eta = self.eta_scheduler.get_variable()

        # probability to activate shielding: annealing to 1 and in the
        # timescale of steps.
        self.epsilon_scheduler = StepLRMargin(
            init_value=cfg.eps, period=cfg.eps_period, threshold=0,
            decay=cfg.eps_decay, end_value=cfg.eps_end, goal_value=1.
        )
        self.eps = self.epsilon_scheduler.get_variable()

        # probability of using backup policy: decaying exponentially to 0 and
        # in the timescale of steps.
        self.rho_scheduler = StepLR(
            init_value=cfg.rho, period=cfg.rho_period, decay=cfg.rho_decay,
            end_value=0.
        )
        self.rho = self.rho_scheduler.get_variable()
        self.rho_traj = self.cfg.rho_traj

        # probability of adding noise to append: decaying exponentially to 0
        # and in the timescale of steps.
        self.use_append_noise = False
        if cfg.use_append_noise:
            self.use_append_noise = True
            self.append_noise_std = torch.tensor([
                cfg.l_x_noise_std, cfg.heading_noise_std
            ], device=self.device)
            self.nu_scheduler = StepLR(
                init_value=cfg.nu, period=cfg.nu_period, decay=cfg.nu_decay,
                end_value=0.
            )
            self.nu = self.nu_scheduler.get_variable()
            self.append_noise_dist = torch.distributions.Normal(
                torch.zeros((2), device=self.device),
                self.nu * self.append_noise_std
            )

        # For saving and removing models
        self.module_all = [self.performance, self.backup]

    @property
    def has_backup(self):
        return True

    def learn(
        self, venv: VecEnvWrapper, env: BaseEnv,
        current_step: Optional[int] = None, vmin: float = -1, vmax: float = 1,
        save_figure: bool = True, plot_figure: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Learns the prior of PAC-Shield-Perf.

        Args:
            venv (VecEnvWrapper):  vectorized environment.
            env (BaseEnv): a single environment for evaluation.
            current_step (Optional[int], optional): current number of training
                steps. Defaults to None.
            vmin (float, optional): minimum value for plotting. Defaults to -1.
            vmax (float, optional): maximum value for plotting. Defaults to 1.
            save_figure (bool, optional): saves figures if True. Defaults to
                True.
            plot_figure (bool, optional): plots figures if True. Defaults to
                False.

        Returns:
            np.ndarray: training loss at every record cycle.
            np.ndarray: training progress at every record cycle.
            np.ndarray: safety violations so far at every record cycle.
            np.ndarray: experienced episodes so far at every record cycle.
        """
        # hyper-parameters
        max_steps = self.cfg.max_steps
        opt_freq = self.cfg.optimize_freq
        num_update_per_opt = self.cfg.update_per_opt
        check_opt_freq = self.cfg.check_opt_freq
        min_step_b4_opt = self.cfg.min_steps_b4_opt
        if self.cfg.check_type == 'all_env':
            num_traj_per_env = self.cfg.num_traj_per_env
            num_rnd_traj = num_traj_per_env * venv.num_task
        else:
            num_rnd_traj = self.cfg.num_validation_trajs
        plot_v = self.cfg.plot_v
        out_folder = self.cfg.out_folder

        # == Main Training ==
        start_learning = time.time()
        train_records = [[], []]
        train_progress = [[], [], [], [], []]
        violation_record = []
        episode_record = []
        cnt_opt = 0
        cnt_opt_period = 0
        cnt_safety_violation = 0
        cnt_num_episode = 0

        # Saves model
        model_folder = os.path.join(out_folder, 'model')
        model_folder_perf = os.path.join(model_folder, 'performance')
        model_folder_backup = os.path.join(model_folder, 'backup')
        model_folder_disc = os.path.join(model_folder_perf, 'disc')
        os.makedirs(model_folder_perf, exist_ok=True)
        os.makedirs(model_folder_backup, exist_ok=True)
        os.makedirs(model_folder_disc, exist_ok=True)
        self.module_folder_all = [model_folder_perf, model_folder_backup]
        save_metric = self.cfg.save_metric

        if save_figure:
            figure_folder_perf = os.path.join(
                out_folder, 'figure', 'performance'
            )
            figure_folder_backup = os.path.join(out_folder, 'figure', 'backup')
            figure_folder_shield = os.path.join(out_folder, 'figure', 'shield')
            os.makedirs(figure_folder_perf, exist_ok=True)
            os.makedirs(figure_folder_backup, exist_ok=True)
            os.makedirs(figure_folder_shield, exist_ok=True)

        if current_step is None:
            self.cnt_step = 0
        else:
            self.cnt_step = current_step
            print("starting from {:d} steps".format(self.cnt_step))

        # Shielding info
        shield_dict = self.cfg.train_shield_dict
        shield_type = shield_dict['type']

        # Resets all envs
        s, task_ids = venv.reset(random_init=False)
        z = self.performance.latent_prior.sample((self.n_envs,))
        z = z.to(self.device)
        _k_x_all = [0] * self.n_envs
        if self.rho_traj:
            use_backup = [self.rng.random() < self.rho for _ in _k_x_all]
            use_perf = [not backup for backup in use_backup]

        while self.cnt_step <= max_steps:
            print(self.cnt_step, end='\r')

            # Setss train modes for all envs
            venv.env_method('set_train_mode')

            # Checks if using backup or performance policy
            if not self.rho_traj:
                if self.get_l_x_ra:
                    use_backup = [
                        self.rng.random() < self.rho and k_x > 2.0
                        for k_x in _k_x_all
                    ]
                else:
                    use_backup = [
                        self.rng.random() < self.rho for _ in _k_x_all
                    ]
                use_perf = [not backup for backup in use_backup]

            # Checks if applying shielding
            apply_shielding = ((self.rng.random() < self.eps)
                               and (self.cfg.use_shielding))

            # Gets append for performance policy
            append_all = venv.get_append(venv.get_attr('_state'))
            if self.use_append_noise:
                append_noise = self.append_noise_dist.sample((self.n_envs,))
                append_all += append_noise

            # Selects action - random if before 1st optimize
            with torch.no_grad():
                a_exec = torch.zeros((self.n_envs, self.action_dim + 1))
                a_exec = a_exec.to(self.device)
                a_perf, _ = self.performance.actor.sample(
                    s[use_perf], append=append_all[use_perf],
                    latent=z[use_perf]
                )
                a_backup, _ = self.backup.actor.sample(
                    s[use_backup], append=append_all[use_backup]
                )
                a_exec[use_perf, :-1] = a_perf
                a_exec[use_perf, -1] = self.performance.act_ind
                a_exec[use_backup, :-1] = a_backup
                a_exec[use_backup, -1] = self.backup.act_ind
            a_shield = a_exec.clone()

            # Saves robot state for reloading
            _prev_states = np.array(venv.get_attr('_state'))

            # Shielding
            if apply_shielding:
                shield_flag, _ = check_shielding(
                    self.backup, shield_dict, s, a_exec, append_all,
                    context_backup=None, state=_prev_states,
                    policy=self.performance, context_policy=z
                )
                if torch.any(shield_flag):
                    a_backup = self.backup.actor(
                        s[shield_flag], append=append_all[shield_flag]
                    ).data
                    a_shield[shield_flag, :-1] = a_backup
                    a_shield[shield_flag, -1] = self.backup.act_ind

                # Resets robot to the previous state before shielding
                if shield_type != 'value':
                    venv.env_method_arg(
                        'reset_robot',
                        [(_prev_state,) for _prev_state in _prev_states]
                    )

            # Applies action - update heading
            s_all, r_all, done_all, info_all = venv.step(a_shield)

            # Gets new append
            append_next_all = venv.get_append(venv.get_attr('_state'))

            for env_ind, (s_, r, done, info) in enumerate(
                zip(s_all, r_all, done_all, info_all)
            ):
                # Add task id to info before storing to replay buffer
                info['task_id'] = task_ids[env_ind]

                # Saves append
                info['append'] = append_all[env_ind].unsqueeze(0)
                info['append_nxt'] = append_next_all[env_ind].unsqueeze(0)

                # Updates k_x
                _k_x_all[env_ind] = info['k_x']

                info['a_perf'] = a_exec[env_ind, :-1].unsqueeze(0)

                # Stores the transition in memory
                self.store_transition(
                    z[env_ind].clone().unsqueeze(0),
                    s[env_ind].unsqueeze(0).to(self.image_device),
                    a_shield[env_ind].unsqueeze(0), r,
                    s_.unsqueeze(0).to(self.image_device), done, info
                )

                # Checks if episode finished
                if done:

                    # Checks if sampling task from replay buffer
                    sample_task_from_buffer = self.rng.random() < self.eta
                    if sample_task_from_buffer:
                        latent, task_id, state = \
                            self.unpack_single_task_from_batch(
                                self.sample_batch(batch_size=1))
                        bs_task = copy.copy(venv.task_all[task_id])

                        # Sets state
                        bs_task['init_state'] = state

                        # Use the same latent as before
                        z[env_ind] = latent

                        # Resets environment with bootstrap task
                        obs, _ = venv.reset_one(env_ind, task=bs_task)

                    else:
                        # Re-sample z
                        z[env_ind] = self.performance.latent_prior.sample().to(
                            self.device
                        )

                        # Resets environment with randomly sampled task
                        obs, task_id = venv.reset_one(
                            env_ind, random_init=False
                        )

                        # Sets distance from init be zero
                        _k_x_all[env_ind] = 0

                    # Updates use_perf/backup
                    if self.rho_traj:
                        use_backup[env_ind] = self.rng.random() < self.rho
                        use_perf[env_ind] = not use_backup[env_ind]

                    # Updates task id
                    task_ids[env_ind] = task_id

                    # Updates observations
                    s_all[env_ind] = obs

                    # Checks safety violation
                    g_x = info['g_x']
                    if g_x > 0:
                        cnt_safety_violation += 1

                    cnt_num_episode += 1
            violation_record.append(cnt_safety_violation)
            episode_record.append(cnt_num_episode)

            # Updatess "prev" states
            s = s_all

            # Optimizes
            if (
                self.cnt_step >= min_step_b4_opt and cnt_opt_period >= opt_freq
            ):
                cnt_opt_period = 0
                loss_perf = np.zeros(5)  # disc
                loss_backup = np.zeros(4)

                # Updates discriminator
                if self.perf_diversity_type == 'state':
                    # Updates discriminator
                    loss_disc = []
                    for _ in range(self.disc_num_update_per_opt):
                        _online_batch = self.sample_batch(
                            batch_size=self.disc_batch_size,
                            recent_size=self.disc_recent_size
                        )
                        online_batch = self.unpack_batch(
                            _online_batch, get_latent=True,
                            get_perf_action=True
                        )
                        loss_disc += [
                            self.performance.update_disc(online_batch)
                        ]
                    loss_disc = np.mean(loss_disc)
                else:
                    loss_disc = 0.  # dummy

                # Updates critic/actor
                for timer in range(num_update_per_opt):
                    batch = self.sample_batch()
                    batch_perf = self.unpack_batch(
                        batch, get_latent=True,
                        get_perf_action=self.action_relabel,
                        get_l_x_ra=self.get_l_x_ra
                    )
                    batch_backup = self.unpack_batch(
                        batch, get_latent=False, get_perf_action=False,
                        get_l_x_ra=self.get_l_x_ra
                    )

                    loss_tp_perf = self.performance.update(
                        batch_perf, timer, update_period=self.update_period
                    )
                    loss_tp_backup = self.backup.update(
                        batch_backup, timer, update_period=self.update_period
                    )
                    for i, l in enumerate(loss_tp_perf):
                        loss_perf[i] += l
                    for i, l in enumerate(loss_tp_backup):
                        loss_backup[i] += l
                loss_perf /= num_update_per_opt
                loss_backup /= num_update_per_opt
                loss_perf[-1] = loss_disc

                # Record
                train_records[0].append(
                    loss_perf
                )  # loss_q, loss_pi, loss_entropy, loss_alpha, loss_disc
                train_records[1].append(
                    loss_backup
                )  # loss_q, loss_pi, loss_entropy, loss_alpha
                if self.cfg.use_wandb:
                    lr = self.performance.critic_optimizer.state_dict(
                    )['param_groups'][0]['lr']
                    wandb.log({
                        "loss_q (Perf)": loss_perf[0],
                        "loss_pi (Perf)": loss_perf[1],
                        "loss_entropy (Perf)": loss_perf[2],
                        "loss_alpha (Perf)": loss_perf[3],
                        "loss_disc (Perf)": loss_perf[4],
                        "learning rate (Perf)": lr,
                        "loss_q (Backup)": loss_backup[0],
                        "loss_pi (Backup)": loss_backup[1],
                        "loss_entropy (Backup)": loss_backup[2],
                        "loss_alpha (Backup)": loss_backup[3],
                        "GAMMA (Backup)": self.backup.gamma
                    }, step=self.cnt_step, commit=False)

                # Re-initializes pb to avoid memory explosion from mesh loading
                # - this will also terminates any trajectories at sampling
                venv.env_method('close_pb')
                s, task_ids = venv.reset(random_init=False, verbose=False)

                # Counts number of optimization
                cnt_opt += 1

                # Checks after fixed number of steps
                if cnt_opt % check_opt_freq == 0:
                    print(
                        '\n  - Safety violations so far: {:d}'.
                        format(cnt_safety_violation)
                    )
                    print(
                        '  - eta={:.2f}, eps={:.2f}, rho={:.2f}'.format(
                            self.eta, self.eps, self.rho
                        )
                    )
                    if self.use_append_noise:
                        print('  - nu={:.2f}'.format(self.nu))

                    # Releases GPU RAM as much as possible
                    torch.cuda.empty_cache()

                    # Sets states for check()
                    sample_states = self.get_check_states(env, num_rnd_traj)
                    if self.cfg.check_type == 'all_env':
                        progress_perf = self.performance.check(
                            venv,
                            self.cnt_step,
                            states=sample_states,
                            check_type='all_env',
                            verbose=True,
                            num_traj_per_env=num_traj_per_env,
                            revert_task=True,
                            # kwargs for env.simulate_trajectories
                            latent_dist=self.performance.latent_prior,
                        )
                        progress_backup = self.backup.check(
                            venv,
                            self.cnt_step,
                            states=sample_states,
                            check_type='all_env',
                            verbose=True,
                            num_traj_per_env=num_traj_per_env,
                            revert_task=True,
                        )
                        progress_perf_shield_value = self.performance.check(
                            venv,
                            self.cnt_step,
                            states=sample_states,
                            check_type='all_env',
                            verbose=True,
                            num_traj_per_env=num_traj_per_env,
                            revert_task=True,
                            # kwargs for env.simulate_trajectories
                            latent_dist=self.performance.latent_prior,
                            shield=True,
                            backup=self.backup,
                            shield_dict=self.cfg.value_shield_dict,
                        )
                    else:
                        progress_perf = self.performance.check(
                            venv,
                            self.cnt_step,
                            states=sample_states,
                            check_type='random',
                            verbose=True,
                            revert_task=True,
                            sample_task=True,
                            num_rnd_traj=num_rnd_traj,
                            # kwargs for env.simulate_trajectories
                            latent_dist=self.performance.latent_prior,
                        )
                        progress_backup = self.backup.check(
                            venv,
                            self.cnt_step,
                            states=sample_states,
                            check_type='random',
                            verbose=True,
                            revert_task=True,
                            sample_task=True,
                            num_rnd_traj=num_rnd_traj,
                        )
                        progress_perf_shield_value = self.performance.check(
                            venv,
                            self.cnt_step,
                            states=sample_states,
                            check_type='random',
                            verbose=True,
                            revert_task=True,
                            sample_task=True,
                            num_rnd_traj=num_rnd_traj,
                            # kwargs for env.simulate_trajectories
                            latent_dist=self.performance.latent_prior,
                            shield=True,
                            backup=self.backup,
                            shield_dict=self.cfg.value_shield_dict,
                        )
                    progress_perf_shield_sim = [0]
                    train_progress[0].append(progress_perf)
                    train_progress[1].append(progress_backup)
                    train_progress[2].append(progress_perf_shield_value)
                    train_progress[3].append(progress_perf_shield_sim)
                    train_progress[4].append(self.cnt_step)
                    if self.cfg.use_wandb:
                        wandb.log({
                            "Success (Perf)": progress_perf[0],
                            "Success (Backup)": progress_backup[0],
                            "Success (Value)": progress_perf_shield_value[0],
                            "cnt_safety_violation": cnt_safety_violation,
                            "cnt_num_episode": cnt_num_episode,
                            "ETA": self.eta,
                            "EPS": self.eps,
                            "RHO": self.rho
                        }, step=self.cnt_step, commit=True)
                        if self.use_append_noise:
                            wandb.log({
                                "NU": self.nu,
                            }, step=self.cnt_step, commit=True)

                    # Saves model
                    if save_metric == 'value':
                        self.save(metric=progress_perf_shield_value[0])
                    else:
                        raise NotImplementedError

                    # Saves training details
                    torch.save({
                        'train_records': train_records,
                        'train_progress': train_progress,
                        'violation_record': violation_record,
                        "episode_record": episode_record,
                    }, os.path.join(out_folder, 'train_details'))

                    if plot_figure or save_figure:
                        self.get_figures(
                            venv, env, plot_v, vmin, vmax, save_figure,
                            plot_figure, figure_folder_perf, plot_backup=True,
                            figure_folder_backup=figure_folder_backup,
                            plot_shield=True, plot_shield_value=True,
                            figure_folder_shield=figure_folder_shield,
                            shield_dict=shield_dict
                        )

                    # Releases GPU RAM as much as possible
                    torch.cuda.empty_cache()

                    # Re-initializes env
                    env.close_pb()
                    env.reset()

            # Counts
            self.cnt_step += self.n_envs
            cnt_opt_period += self.n_envs

            # Updates gamma, lr etc.
            for _ in range(self.n_envs):
                self.performance.update_hyper_param()
                self.backup.update_hyper_param()
                self.eta_scheduler.step()
                self.epsilon_scheduler.step()
                self.rho_scheduler.step()
                if self.use_append_noise:
                    self.nu_scheduler.step()
            if self.use_append_noise:
                self.nu = self.nu_scheduler.get_variable()
                self.append_noise_dist = torch.distributions.Normal(
                    torch.zeros((2), device=self.device),
                    self.nu * self.append_noise_std
                )
            self.eta = self.eta_scheduler.get_variable()
            self.eps = self.epsilon_scheduler.get_variable()
            self.rho = self.rho_scheduler.get_variable()

        self.save(force_save=True)
        end_learning = time.time()
        time_learning = end_learning - start_learning
        print('\nLearning: {:.1f}'.format(time_learning))

        for i, tp in enumerate(train_records):
            train_records[i] = np.array(tp)
        for i, tp in enumerate(train_progress):
            train_progress[i] = np.array(tp)
        violation_record = np.array(violation_record)
        episode_record = np.array(episode_record)
        return train_records, train_progress, violation_record, episode_record
