#import statements
import tensorflow as tf
from os import makedirs
from os.path import join as ospath_join

from PhysicalEnvironment.SystemManager import SystemManager
from PhysicalEnvironment.StateTrajectory import Trajectory
from Visualization.Logger import LossLogger, RewardPenaltyLogger
from .Controller import Controller
from .ReplayBuffer import ReplayBuffer
from .Rewards import RewardBase, ControlSignalPenalizerBase, RewardPlusPenalizer
from .NetBlocks import DenseNet





class DDPG_Alfa(Controller):
    """
    First implementation of the DDPG algorithm 'Algorithm 7' in chapter 3 of Rita Laezzas publication.
    """

    def __init__(self, actor_lyrs, critic_lyrs, dense_kwargs, num_state_vars=3, num_control_vars=2, name="DDPG_Alfa"):
        self._n_svars = num_state_vars
        self._n_cvars = num_control_vars
        self._reached_goal = tf.Variable(0.0, dtype=tf.float32, shape=(), trainable=False)
        self._name = name
        
        if actor_lyrs[-1] != num_control_vars: actor_lyrs.append(num_control_vars)
        if critic_lyrs[-1] != 1: critic_lyrs.append(1)

        actor_indims = num_state_vars
        critic_indims = num_state_vars + num_control_vars

        self._actor:DenseNet = DenseNet(actor_lyrs, actor_indims, **dense_kwargs)
        self._critic:DenseNet = DenseNet(critic_lyrs, critic_indims, **dense_kwargs)
        self._target_actor:DenseNet = DenseNet(actor_lyrs, actor_indims, **dense_kwargs)
        self._target_critic:DenseNet = DenseNet(critic_lyrs, critic_indims, **dense_kwargs)
        
        for i in range(len(self._actor.var_cache)):
            self._target_actor.var_cache[i].assign(self._actor.var_cache[i], read_value=False)
        for i in range(len(self._critic.var_cache)):
            self._target_critic.var_cache[i].assign(self._critic.var_cache[i], read_value=False)
    
    def __str__(self):
        return self._name
    
    def save_model(self, save_dir):
        act_folder = ospath_join(save_dir, "actor")
        makedirs(act_folder, exist_ok=True)
        crit_folder = ospath_join(save_dir, "critic")
        makedirs(crit_folder, exist_ok=True)
        targ_act_folder = ospath_join(save_dir, "target_actor")
        makedirs(targ_act_folder, exist_ok=True)
        targ_crit_folder = ospath_join(save_dir, "target_critic")
        makedirs(targ_crit_folder, exist_ok=True)
        self._actor.save_network(act_folder, "actor")
        self._critic.save_network(crit_folder, "critic")
        self._target_actor.save_network(targ_act_folder, "target_actor")
        self._target_critic.save_network(targ_crit_folder, "target_critic")
    
    def config_training(self, num_episodes=10, episode_max_length=100, batch_size=16,
                        gamma_decay=0.8, lr_critic=0.04, lr_actor=0.06, tau_targets=0.05):
        
        self._ne = tf.constant(num_episodes, dtype=tf.int32)
        self._ep_l = tf.constant(episode_max_length, dtype=tf.int32)
        self._bs = tf.constant(batch_size, dtype=tf.int32)
        self._bsdiv = tf.cast(self._bs, tf.float32)
        self._gd = tf.constant(gamma_decay, dtype=tf.float32)
        self._tau = tf.constant(tau_targets, dtype=tf.float32)

        self._actor_optimizer = tf.optimizers.SGD(learning_rate=lr_actor, momentum=0.1, name="ActorOptimizer")
        self._critic_optimizer = tf.optimizers.SGD(learning_rate=lr_critic, momentum=0.1, name="CriticOptimizer")

    def train(self, sm:SystemManager, random_action_process:callable, reward_fn:RewardBase,
              replay_buffer:ReplayBuffer):
        if not self._ne: print("Error! Training not configured!"); return

        gd = self._gd

        idx_s0 = 0
        idx_a = idx_s0 + self._n_svars
        idx_r = idx_a + self._n_cvars
        idx_s1 = idx_r + 1
        idx_g = idx_s1 + self._n_svars

        for ep in tf.range(self._ne):

            sm.reset_environment()

            s = sm.read_polar_state()
            self._reached_goal.assign(0.0)
            reward_fn.reset(s)

            for t in tf.range(self._ep_l):
                
                #Continue trajectory:
                c = random_action_process(self._actor.call_on_single_input(s))
                s_incs, is_goal, is_oob = sm.tick(c)
                if is_goal: self._reached_goal.assign(1.0, read_value=False)
                s_new = sm.read_polar_state()

                r = reward_fn(old_state=s,action=c,new_state=s_new, state_incs=s_incs, is_goal=is_goal, is_oob=is_oob)
                
                replay_buffer.add_transition(s, c, r, s_new, self._reached_goal.read_value())
                s = s_new

                #Sample minibatch and update nets:
                batch = replay_buffer.sample_minibatch(self._bs)

                tr_s0 = batch[:,idx_s0:idx_a]
                tr_a = batch[:,idx_a:idx_r]
                tr_r = batch[:,idx_r:idx_s1]
                tr_s1 = batch[:,idx_s1:idx_g]
                tr_g = batch[:,idx_g:]

                gt = tf.GradientTape(watch_accessed_variables=False)

                ## critic:
                
                a = self._target_actor(tr_s1)
                y = tr_r + gd * (1 - tr_g) * self._target_critic(tf.concat([tr_s1, a], axis=1))

                with gt:
                    gt.watch(self._critic.var_cache)
                    loss = tf.reduce_sum(tf.square(y - self._critic(tf.concat([tr_s0, tr_a], axis=1))))
                    loss = tf.divide(loss, self._bsdiv)
                
                crit_grads = gt.gradient(loss, self._critic.var_cache)
                for grad in crit_grads:
                    tf.debugging.check_numerics(grad, message='NaN in critics grads!')
                #gradient descent step:
                self._critic_optimizer.apply_gradients(zip(crit_grads, self._critic.var_cache))

                ## actor:
                with gt:
                    gt.watch(self._actor.var_cache)
                    loss = tf.reduce_sum(self._critic(tf.concat([tr_s0, self._actor(tr_s0)], axis=1)))
                    loss = - tf.divide(loss, self._bsdiv)
                
                act_grads = gt.gradient(loss, self._actor.var_cache)
                for grad in act_grads:
                    tf.debugging.check_numerics(grad, message='NaN in actors grads!')
                #gradient ascent step (because loss was made to change sign):
                self._actor_optimizer.apply_gradients(zip(act_grads, self._actor.var_cache))

                ## target nets:
                for i, var in enumerate(self._target_actor.var_cache):
                    var.assign_add(self._tau * (self._actor.var_cache[i] - var), read_value=False)
                
                for i, var in enumerate(self._target_critic.var_cache):
                    var.assign_add(self._tau * (self._critic.var_cache[i] - var), read_value=False)
                
                #Check for loop-end:
                if self._reached_goal == 1.0 or is_oob:
                    break

    def get_control_signal(self, environment_state: tf.Tensor):
        return self._actor.call_on_single_input(environment_state)
    
    #def get_critic_value(self, state, action):
    #    return self._critic(tf.concat([state, action]))

    #def get_target_critic_value(self, state, action):
    #    return self._target_critic(tf.concat([state, action]))
    
    #def get_target_control_signal(self, environment_state: tf.Tensor):
    #    return self._target_actor(environment_state)


class DDPG_Beta(DDPG_Alfa):

    def __init__(self, *args, **kwargs):
        if not "name" in kwargs.keys():
            kwargs["name"] = "DDPG_Beta"
        super().__init__(*args, **kwargs)
    
    def config_training(self, buffer_sampling_n_newest_transitions=0, **superkwargs):
        self._buffer_n_newest = buffer_sampling_n_newest_transitions
        super().config_training(**superkwargs)

    def train(self, sm: list[SystemManager], random_action_process: callable, reward_fn: RewardBase,
              replay_buffer: ReplayBuffer, loss_logger:LossLogger,
              penalizer:ControlSignalPenalizerBase=None, rew_pen_logger:RewardPenaltyLogger=None,
              training_trajlog:list[Trajectory]=None):
        if not self._ne: print("Error! Training not configured!"); return

        gd = self._gd
        #do_pen_step = penalizer is not None #This is about applying penalty directly to actor during trajectory progression
        log_rew_pen = rew_pen_logger is not None and type(reward_fn) == RewardPlusPenalizer
        log_trajs = training_trajlog is not None
        if log_trajs:
            if not self._ne == len(training_trajlog):
                print("Warning! Training StateTrajectry loglist length not equal to number of episodes! Not logging and plotting training trajectories.")
                log_trajs = False

        idx_s0 = 0
        idx_a = idx_s0 + self._n_svars
        idx_r = idx_a + self._n_cvars
        idx_s1 = idx_r + 1
        idx_g = idx_s1 + self._n_svars

        reached_goal_counter = 0

        for ep in tf.range(self._ne):

            for sm_i in sm:
                sm_i.reset_environment()

            self._reached_goal.assign(0.0)
            #reward_fn.reset(s)

            for t in tf.range(self._ep_l):
                
                #Continue trajectory:
                counter = tf.constant(0, dtype=tf.int32, shape=())
                for sm_i in sm:
                    if sm_i.is_goal() or sm_i.is_out_of_bounds(): continue;
                    counter += 1
                    s = sm_i.read_polar_state()
                    c = random_action_process(self._actor.call_on_single_input(s))
                    if log_trajs: time_old = sm_i.time
                    s_incs, is_goal, is_oob = sm_i.tick(c)
                    s_new = sm_i.read_polar_state()
                    if log_rew_pen:
                        rew = reward_fn.get_reward_evaluation(old_state=s, action=c, new_state=s_new, state_incs=s_incs, is_goal=is_goal, is_oob=is_oob)
                        pen = reward_fn.get_penalty_evaluation(old_state=s, action=c, new_state=s_new, state_incs=s_incs, is_goal=is_goal, is_oob=is_oob)
                        rew_pen_logger.add_reward(rew)
                        rew_pen_logger.add_penalty(pen)
                    r = reward_fn(old_state=s, action=c, new_state=s_new, state_incs=s_incs, is_goal=is_goal, is_oob=is_oob)
                    if log_trajs:
                        training_trajlog[ep].add_point(s, c, r, time_old)
                        if is_oob:
                            training_trajlog[ep].add_point(s_new, tf.constant(0.0, shape=c.shape), 0.0, sm_i.time)
                    if is_goal:
                        self._reached_goal.assign(1.0, read_value=False)
                        reached_goal_counter += 1
                        if log_trajs:
                            training_trajlog[ep].reached_goal = True
                            training_trajlog[ep].add_point(s_new, tf.constant(0.0, shape=c.shape), 0.0, sm_i.time)
                    replay_buffer.add_transition(s, c, r, s_new, self._reached_goal)
                    self._reached_goal.assign(0.0, read_value=False)
                
                if tf.equal(counter, 0): break

                #Sample minibatch and update nets:
                for i in tf.range(0, tf.minimum(counter, t + 1)):
                    batch = replay_buffer.sample_minibatch(self._bs, include_n_newest_transitions=self._buffer_n_newest)

                    tr_s0 = batch[:,idx_s0:idx_a]
                    tr_a = batch[:,idx_a:idx_r]
                    tr_r = batch[:,idx_r:idx_s1]
                    tr_s1 = batch[:,idx_s1:idx_g]
                    tr_g = batch[:,idx_g:]

                    gt = tf.GradientTape(watch_accessed_variables=False)

                    ## critic:
                
                    a = self._target_actor(tr_s1)
                    targ_crit_vals = self._target_critic(tf.concat([tr_s1, a], axis=1))
                    y = tr_r + gd * (1 - tr_g) * targ_crit_vals

                    with gt:
                        gt.watch(self._critic.var_cache)
                        crit_vals = self._critic(tf.concat([tr_s0, tr_a], axis=1))
                        crit_loss = tf.sqrt(tf.reduce_sum(tf.square(y - crit_vals)))
                        crit_loss = tf.divide(crit_loss, tf.cast(batch.shape[0], dtype=tf.float32))
                
                    crit_grads = gt.gradient(crit_loss, self._critic.var_cache)
                    for grad in crit_grads:
                        tf.debugging.check_numerics(grad, message='NaN in critics grads! episode:{epi}, time step:{t_step}'.format(epi=ep, t_step=t))
                        
                    #gradient descent step:
                    self._critic_optimizer.apply_gradients(zip(crit_grads, self._critic.var_cache))
                    loss_logger.add_critic_loss(crit_loss)

                    ## actor:
                    with gt:
                        gt.watch(self._actor.var_cache)
                        act_loss = tf.reduce_sum(self._critic(tf.concat([tr_s0, self._actor(tr_s0)], axis=1)))
                        act_loss = - tf.divide(act_loss, tf.cast(batch.shape[0], dtype=tf.float32))
                
                    act_grads = gt.gradient(act_loss, self._actor.var_cache)
                    for grad in act_grads:
                        tf.debugging.check_numerics(grad, message='NaN in actors grads! episode:{epi}, time step:{t_step}'.format(epi=ep, t_step=t))

                    #gradient ascent step (because loss was made to change sign):
                    self._actor_optimizer.apply_gradients(zip(act_grads, self._actor.var_cache))
                    loss_logger.add_actor_loss(act_loss)

                    ## target nets:
                    for i, var in enumerate(self._target_actor.var_cache):
                        var.assign_add(self._tau * (self._actor.var_cache[i] - var), read_value=False)
                
                    for i, var in enumerate(self._target_critic.var_cache):
                        var.assign_add(self._tau * (self._critic.var_cache[i] - var), read_value=False)
        
        print("Reached goal " + str(reached_goal_counter) + " times during training session.")