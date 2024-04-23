#import statements
import numpy as np
import tensorflow as tf
from os import makedirs
from os.path import join as ospath_join

from Control.Controller import Controller
from Control.DDPG import DDPG_Alfa, DDPG_Beta
from Control.MAicardiController import MAicardiController
from Control.ReplayBuffer import AllRandomBuffer, ReplayBuffer
from PhysicalEnvironment.GoalAndOOBFunctions import alfa_goal, norm_goal, good_goal, norm_goal2, oob_on_e_distance
from PhysicalEnvironment.StateInitializers import AlfaStateInitializer, ConstantInitializer, NormalizedRandomInitializer
from PhysicalEnvironment.StateTrajectory import NormalizedTrajectory, Trajectory
from PhysicalEnvironment.SystemManager import SystemManager
from PhysicalEnvironment.SystemState import NormalizedSystemState, SystemState
from PhysicalEnvironment.SystemDynamics import NormalizedSystemDynamics, SystemDynamics
from Control.Rewards import ComparativeLyapunovReward, ComparativeLocallyNormalizedLyapunovReward, NegativeLyapunovDerivative, PositiveLyapunovDerivative, IndividualAbsLyapunovDerivative, NormalizedLyapunovReward
from Control.Rewards import RewardPlusPenalizer, NormalRewardOrGoalRandOOBR, TanhAppliedReward, RewardAlfa, NoReward
from Control.Rewards import QuadraticPenalizer, ControlSignalPenalizerBase as LinearPenalizer, LogarithmicPenalizer
from Control.Rewards import LyapDeriv_eq11, LyapDeriv_eq11_comparison, LyapDeriv_eq11_and_thetaterm
from Simulation.SimulationLoops import alfa_simulation, controller_test_alg1
from Simulation.TrainingManager import TrainingManager
from Visualization.Visualizer import Visualizer
from Visualization.Logger import LossLogger, RewardPenaltyLogger
from Control.ActionExploration import exploration_with_probability, gauss_add_withstddev, gaussian_addition, no_exploration, RandGenHolder, gauss_add_relative_stddev


def get_save_dir():
    #d = r"D:\Users\matsh\Documents\Examensarbeten\Kandidatarbete i Trajectory Planning\Results\MAicardiController\Test_0"
    #d = r"D:\Users\matsh\Documents\Examensarbeten\Kandidatarbete i Trajectory Planning\Results\DDPG\Bugtesting"
    #d = r"D:\Users\matsh\Documents\Examensarbeten\Kandidatarbete i Trajectory Planning\Results\DDPG\Norm1Testing"
    #d = r"D:\Users\matsh\Documents\Examensarbeten\Kandidatarbete i Trajectory Planning\Results\DDPG\Norm_And_Pen1_Testing"
    #d = r"D:\Users\matsh\Documents\Examensarbeten\Kandidatarbete i Trajectory Planning\Results\DDPG\Penald1_Testing"
    #d = r"D:\Users\matsh\Documents\Examensarbeten\Kandidatarbete i Trajectory Planning\Results\DDPG\Norm_And_Pen2_Testing"
    #d = r"D:\Users\matsh\Documents\Examensarbeten\Kandidatarbete i Trajectory Planning\Results\DDPG\Norm_And_Pen_batchsm_Testing"
    #d = r"D:\Users\matsh\Documents\Examensarbeten\Kandidatarbete i Trajectory Planning\Results\DDPG\CompLclNormLyapuR_Testing"
    #d = r"D:\Users\matsh\Documents\Examensarbeten\Kandidatarbete i Trajectory Planning\Results\DDPG\CompLclNormLyp_QuadPen_Testing"
    #d = r"D:\Users\matsh\Documents\Examensarbeten\Kandidatarbete i Trajectory Planning\Results\DDPG\Pen_No_R_Testing"
    #d = r"D:\Users\matsh\Documents\Examensarbeten\Kandidatarbete i Trajectory Planning\Results\DDPG\Tanh_CompLclNormLyp_QuadPen_Testing"
    #d = r"D:\Users\matsh\Documents\Examensarbeten\Kandidatarbete i Trajectory Planning\Results\DDPG\TanhCLNLyp_LinPen_Testing"
    #d = r"D:\Users\matsh\Documents\Examensarbeten\Kandidatarbete i Trajectory Planning\Results\DDPG\TanhCLNLyp_LogPen_Testing"
    #d = r"D:\Users\matsh\Documents\Examensarbeten\Kandidatarbete i Trajectory Planning\Results\DDPG\Tanh_CompLclNormLyp_LinPen_Testing"
    #d = r"D:\Users\matsh\Documents\Examensarbeten\Kandidatarbete i Trajectory Planning\Results\DDPG\vis_test"
    #d = r"D:\Users\matsh\Documents\Examensarbeten\Kandidatarbete i Trajectory Planning\Results\DDPG\vis_test2"
    #d = r"D:\Users\matsh\Documents\Examensarbeten\Kandidatarbete i Trajectory Planning\Results\DDPG\ClN_Pen_Test1"
    #d = r"D:\Users\matsh\Documents\Examensarbeten\Kandidatarbete i Trajectory Planning\Results\DDPG\tanhClN_Pen_Test1"
    #d = r"D:\Users\matsh\Documents\Examensarbeten\Kandidatarbete i Trajectory Planning\Results\DDPG\vis_test3"
    #d = r"D:\Users\matsh\Documents\Examensarbeten\Kandidatarbete i Trajectory Planning\Results\DDPG_Single_Goal\Test2_LypDeriv"
    #d = r"D:\Users\matsh\Documents\Examensarbeten\Kandidatarbete i Trajectory Planning\Results\DDPG_Single_Goal\Test3_LypDeriv"
    #d = r"D:\Users\matsh\Documents\Examensarbeten\Kandidatarbete i Trajectory Planning\Results\DDPG_Single_Goal\Test4_LypDeriv"
    #d = r"D:\Users\matsh\Documents\Examensarbeten\Kandidatarbete i Trajectory Planning\Results\DDPG_Single_Goal\Test5_LypDeriv"
    #d = r"D:\Users\matsh\Documents\Examensarbeten\Kandidatarbete i Trajectory Planning\Results\DDPG_Single_Goal\Test6_LypDeriv"
    #d = r"D:\Users\matsh\Documents\Examensarbeten\Kandidatarbete i Trajectory Planning\Results\DDPG_Single_Goal\Test7_LypDeriv"
    #d = r"D:\Users\matsh\Documents\Examensarbeten\Kandidatarbete i Trajectory Planning\Results\DDPG_Single_Goal\Test8_LypDeriv"
    #d = r"D:\Users\matsh\Documents\Examensarbeten\Kandidatarbete i Trajectory Planning\Results\DDPG_Single_Goal\Test9_LypDeriv"
    #d = r"D:\Users\matsh\Documents\Examensarbeten\Kandidatarbete i Trajectory Planning\Results\DDPG_Single_Goal\Test10_LypDeriv"
    #d = r"D:\Users\matsh\Documents\Examensarbeten\Kandidatarbete i Trajectory Planning\Results\DDPG_Single_Goal\Test11_LypDeriv"
    #d = r"D:\Users\matsh\Documents\Examensarbeten\Kandidatarbete i Trajectory Planning\Results\DDPG_Single_Goal\Test12_LypDeriv"
    #d = r"D:\Users\matsh\Documents\Examensarbeten\Kandidatarbete i Trajectory Planning\Results\DDPG_Single_Goal\Test13_LypDeriv"
    d = r"D:\Users\matsh\Documents\Examensarbeten\Kandidatarbete i Trajectory Planning\Results\DDPG_Single_Goal\Test14_LypDeriv"
    makedirs(d, exist_ok=True)
    return d

def write_configuration(dir, config_dict:dict):
    with open(ospath_join(dir, "config.txt"), "w") as f:
        for key, val in config_dict.items():
            f.write(key + "\n    " + str(val) + "\n\n")

def leaky_relu(alpha):
    def act(*args, **kwargs):
        return tf.keras.activations.relu(*args, alpha=alpha, **kwargs)
    return act


def configuration_alfa():
    save_dir = get_save_dir()

    config = {"dt" : 0.05,
              "epsilon" : tf.constant(0.001, shape=(), dtype=tf.float32),
              "episode_count" : 5,
              "episode_max_length" : 300,
              "reward_function" : NoReward(),
              "buffer_minibatch_size" : 10,
              "gamma" : 0.9,
              "visualizer_n_veh_markers" : 5,
              "fig_name" : "AlfaTest"}
    
    MAicardiConfig = {"k" : 2.0,
                      "gamma" : 1.0,
                      "h" : 2.0}
    
    controller_info = {"Name" : "M.Aicardi-alfa",
                       "$k$" : MAicardiConfig["k"],
                       "$\u03B3$" : MAicardiConfig["gamma"],
                       "$h$" : MAicardiConfig["h"]}
    
    simulation_info = {"$\u03B4t$" : config["dt"],
                       "$\u03B5$" : tf.get_static_value(config["epsilon"]),
                       "Reward Function" : str(config["reward_function"])}
    
    config["controller_info"] = controller_info
    config["simulation_info"] = simulation_info

    system_manager = SystemManager(config["dt"], config["epsilon"], AlfaStateInitializer())
    controller = MAicardiController(config["epsilon"], **MAicardiConfig)
    visualizer = Visualizer(save_dir, config["visualizer_n_veh_markers"])

    return system_manager, controller, visualizer, config

def configuration_DDPG_config_1():

    dt = 0.03
    epsilon = 0.005

    n_state_vars = 3
    n_control_vars = 2
    
    #actor_lyrs = [5, 6, 8, 8, 6, 5, n_control_vars]
    #critic_lyrs = [5, 6, 8, 8, 6, 5, 1]
    actor_lyrs = [6, 8, 10, 10, 8, 8, n_control_vars]
    critic_lyrs = [6, 8, 10, 10, 8, 8, 1]
    #actor_lyrs = [12, 18, 32, 32, 32, 24, 12, n_control_vars]
    #critic_lyrs = [12, 18, 32, 32, 32, 24, 12, 1]

    n_vehicle_plot_markers = 8

    random_seed = 69
    tf.random.set_seed(random_seed)

    #s_initer = AlfaStateInitializer(e_min=8.0, e_max=12.0, seed=random_seed)
    #s_initer = NormalizedRandomInitializer(seed=2*random_seed)
    s_initer = ConstantInitializer([1.0, -0.2, -0.25])
    system_state_class = NormalizedSystemState
    system_dynamics_class = NormalizedSystemDynamics
    avg_e0 = 1.0
    angle_max = 1.0
    trajectory_class = NormalizedTrajectory
    u_max = 1.2
    w_max = 1.5
    
    train_with_sm_batch = False
    sm_trainingbatch_count = 6
    if not train_with_sm_batch: sm_trainingbatch_count = 1
    rep_sample_batch_size = 12
    repbuffer_reset_keep_fraction = 0.4
    n_training_episodes = 60
    episode_length = 100
    n_sessions = 4
    buffer_force_include_n_newest = 3
    
    controller_class = DDPG_Beta
    #activation_fn = leaky_relu(alpha=0.05)
    #activation_fn_name = "LeakyReLU alpha=0.05"
    activation_fn = tf.keras.activations.elu
    activation_fn_name = "ELU"

    #exploration = exploration_with_probability(0.05, gauss_add_withstddev((u_max + w_max)/3.0))
    #explore_str = "GaussAdd p=0.05"
    #exploration = gauss_add_relative_stddev(fraction=0.2)
    #explore_str = "GaussNoise relative strenght=0.2"
    exploration = gauss_add_withstddev(0.1 * (u_max + w_max) / 2.0)
    explore_str = "GaussNoise stddev=0.1*mean((u,w)_max)"

    def goal_callable(*args):
        return norm_goal2(*args, threshold=0.1)
    #goal_fn = good_goal
    #goal_fn = norm_goal
    goal_fn = goal_callable
    #goal_fn_name = "Good-Goal"
    goal_fn_name = "Norm-G<0.1"
    goal_reward = float(episode_length) / 2.0
    e_max_dist = 2.0
    def oob_callable(*args):
        return oob_on_e_distance(*args, max_dist=e_max_dist)
    oob_fn = oob_callable
    oob_reward = - 10.0  #- float(episode_length) / 3.0
    oob_fn_name = "e-maxdist=" + str(e_max_dist)

    #model_name = "DDPG CompLclNrmLyp & QuadPen"
    #file_name = "DDPG_CompLclNrmLyp_QuadPen"
    #model_name = "DDPG QuadPen NoReward"
    #file_name = "DDPG_QuadPen_NoReward"
    #model_name = "DDPG TanhCLNLyp & LinPen"
    #file_name = "DDPG_TanhCLNLyp_LinPen"
    #model_name = "DDPG TanhCLNLyp LogPen"
    #file_name = "DDPG_TanhCLNLyp_LogPen"
    #model_name = "DDPG_beta tanhClN_LinPen"
    #file_name = "DDPG_beta_tanhClN_LinPen"
    #model_name = "DDPG_beta SingleGoal LypDer"
    #file_name = "DDPG_beta_SingleGoal_LypDer"
    model_name = "DDPG_beta SingleGoal eq11LypDer"
    file_name = "DDPG_beta_SingleGoal_eq11LypDer"

    log_training_trajectories = True
    if log_training_trajectories:
        train_traj_dir = ospath_join(get_save_dir(), "Training_Trajectories")
        makedirs(train_traj_dir, exist_ok=True)

    dense_kwargs = {
        "weights_initializer" : tf.initializers.LecunNormal(3*random_seed),#tf.initializers.HeUniform(3*random_seed),
        "bias_initializer" : tf.initializers.Zeros(),
        "activation_fn" : activation_fn
        }
    
    model_training_kwargs = {
        "num_episodes" : n_training_episodes,
        "episode_max_length" : episode_length,
        "batch_size" : rep_sample_batch_size,
        "gamma_decay" : 0.9,
        "lr_critic" : 0.003,
        "lr_actor" : 0.003,
        "tau_targets" : 0.01,
        "buffer_sampling_n_newest_transitions" : buffer_force_include_n_newest
    }

    #penalizer_kwargs = {
    #    "e0" : avg_e0,
    #    "alfa_theta_max" : angle_max,
    #    "dt" : dt,
    #    "p1" : 50.0,
    #    "p2" : 40.0,
    #    "p3" : 1.0,
    #    "lw" : 1.0
    #}
    penalizer_kwargs = {
        "e0" : avg_e0,
        "ang_max" : angle_max,
        "dt" : dt,
        "u_max" : u_max,
        "w_max" : w_max,
        "L_mid" : 1.0,
        "L_upper_bound" : 1.0,
        "L_lower_bound" : 0.5,
        "cost_scaling": 1.0
    }

    rew_scale = 1.0
    pen_scale = 1.0

    penalizer = None
    #rewarder = NormalizedLyapunovReward(lamb=0.8, h=0.7) #rewards needing reset(s) does not work with DDPG_Beta!!
    #rewarder = ComparativeLyapunovReward(dt=dt, lamb=0.5, h=0.7)
    #rewarder = ComparativeLocallyNormalizedLyapunovReward(dt=dt, lamb=0.7, h=0.5, epsilon=epsilon)
    #rewarder = NoReward()
    #rewarder = NegativeLyapunovDerivative(dt=dt, lamb=0.7, h=0.5)
    #rewarder = PositiveLyapunovDerivative(dt=dt, lamb=0.7, h=0.5)
    #rewarder = IndividualAbsLyapunovDerivative(dt=dt, lamb=0.7, h=0.5)
    #rewarder = LyapDeriv_eq11(lambdagamma=1.0, k=1.0)
    rewarder = LyapDeriv_eq11_and_thetaterm(lambdagamma=1.0, k=2.0, k2=3.0)
    #rewarder = TanhAppliedReward(rewarder, scale=np.sqrt(2), threshold_reward=10.0)
    rewarder = NormalRewardOrGoalRandOOBR(rewarder, goal_R=goal_reward, oob_R=oob_reward)
    #penalizer = LinearControlSignalPenalizer(**penalizer_kwargs)
    #penalizer = QuadraticPenalizer(**penalizer_kwargs)
    penalizer = QuadraticPenalizer(**penalizer_kwargs)
    #penalizer = LogarithmicPenalizer(**penalizer_kwargs)
    #penalizer = TanhAppliedReward(penalizer, scale=3.0, threshold_reward=penalizer.penalty_on_Lupp_plus_Lmid())
    reward_fn = RewardPlusPenalizer(rewarder, penalizer, rew_scale, pen_scale)
    #reward_fn = rewarder
    if penalizer is not None:
        penalizer_description = {"Penalizer as String" : str(penalizer),
                                 "Pen on Lupp + 1xL" : penalizer.penalty_on_Lupp_plus_Lmid().numpy(),
                                 "Pen on ZeroSignal" : penalizer.penalty_on_Zero_Signal().numpy(),
                                 "p1" : penalizer.p1.numpy(),
                                 "p2" : penalizer.p2.numpy(),
                                 "lu" : penalizer.lu.numpy(),
                                 "lw" : penalizer.lw.numpy()}
    else:
        penalizer_description = "No Penalizer"

    expected_trans = model_training_kwargs["num_episodes"] * model_training_kwargs["episode_max_length"] * sm_trainingbatch_count
    
    #rep_B = ReplayBuffer(num_expected_transitions = expected_trans,
    #                     expected_batch_size = model_training_kwargs["batch_size"],
    #                     num_state_vars = n_state_vars, num_control_vars = n_control_vars)
    rep_B = AllRandomBuffer(seed=random_seed*1000, num_expected_transitions = expected_trans,
                            expected_batch_size = model_training_kwargs["batch_size"],
                            num_state_vars = n_state_vars, num_control_vars = n_control_vars)
    
    tm_kwargs = {
        "random_action_process" : exploration,
        "replay_buffer" : rep_B,
        "reward_fn" : reward_fn,
        "evaluation_kwargs" : {"num_eps" : 1, "ep_length" : model_training_kwargs["episode_max_length"], "trajectory_type" : trajectory_class},
        "evaluation_alg" : controller_test_alg1,
        "n_sessions" : n_sessions,
        "plot_session_rate" : 1,
        "repbuffer_reset_keep_fraction" : repbuffer_reset_keep_fraction
    }
    tm_kwargs["loss_logger"] = LossLogger(get_save_dir(), expected_num_logs=expected_trans,
                                          fig_name="Losses " + model_name, runavg_interval=model_training_kwargs["batch_size"])
    tm_kwargs["rew_pen_logger"] = RewardPenaltyLogger(get_save_dir(), expected_num_logs=expected_trans,
                                                      fig_name="Rewards and Penalties " + model_name, runavg_interval=sm_trainingbatch_count*5)
    if log_training_trajectories:
        tm_kwargs["training_trajs"] = [trajectory_class(expected_total_points=model_training_kwargs["episode_max_length"], num_state_vars=n_state_vars, num_control_vars=n_control_vars)
                                       for i in range(model_training_kwargs["num_episodes"])]
        tm_kwargs["visualizer"] = Visualizer(train_traj_dir, n_vehicle_plot_markers)
    RandGenHolder.set_generator_seed(4*random_seed)

    system_manager:list[SystemManager] = [None]*sm_trainingbatch_count
    for i in range(sm_trainingbatch_count):
        system_manager[i] = SystemManager(dt, epsilon, goal_fn, oob_fn, state_initializer=s_initer,
                                   state_class=system_state_class, dynamics_class=system_dynamics_class)
    
    controller = controller_class(actor_lyrs, critic_lyrs, dense_kwargs,
                                  num_state_vars=n_state_vars, num_control_vars=n_control_vars,
                                  name=model_name)
    controller.config_training(**model_training_kwargs)
    
    training_manager = TrainingManager(**tm_kwargs)

    visualizer = Visualizer(get_save_dir(), n_vehicle_plot_markers)

    controller_info = {
        "Name" : model_name,
        "N lyrs" : len(actor_lyrs),
        "N prms" : controller._actor.n_params,
        "Act fn" : activation_fn_name,
    }
    
    simulation_info = {
        "N ep" : model_training_kwargs["num_episodes"] * tm_kwargs["n_sessions"],
        "$\u03B4t$" : dt,
        "$\u03B5$" : epsilon,
    }
    if train_with_sm_batch: simulation_info["N train SMs"] = str(sm_trainingbatch_count)

    training_info = {
        "$\u03B3$" : model_training_kwargs["gamma_decay"],
        "lr" : (model_training_kwargs["lr_critic"] + model_training_kwargs["lr_actor"]) / 2,
        "$\u03C4$" : model_training_kwargs["tau_targets"],
        "Explore" : explore_str
    }
    
    reward_info = {
        "R fn" : str(tm_kwargs["reward_fn"]),
        "G fn" : goal_fn_name,
        "OOB fn" : oob_fn_name,
    }

    plot_kwargs = {
        "fig_name" : model_name,
        "file_name" : file_name,
        "controller_info" : controller_info,
        "simulation_info" : simulation_info,
        "training_info" : training_info,
        "reward_info" : reward_info
    }

    log_config = {"Name" : model_name,
                  "num [s, c]" : [n_state_vars, n_control_vars],
                  "Actor Layers" : actor_lyrs,
                  "Critic Layers" : critic_lyrs,
                  "System state class" : str(system_state_class),
                  "Dynamics class" : str(system_dynamics_class),
                  "Controller class" : str(controller_class),
                  "State Initializer class" : str(s_initer),
                  "Random seed" : random_seed,
                  "dt" : dt,
                  "epsilon" : epsilon,
                  "n sessions" : tm_kwargs["n_sessions"],
                  "evaluation kwargs" : tm_kwargs["evaluation_kwargs"],
                  "controller info" : controller_info,
                  "dense kwargs" : dense_kwargs,
                  "exploration" : explore_str,
                  "reward info" : reward_info,
                  "penalizer kwargs" : penalizer_kwargs,
                  "Penalizer description" : penalizer_description,
                  "model training kwargs" : model_training_kwargs,
                  "Replay Buffer" : str(tm_kwargs["replay_buffer"])}
    write_configuration(get_save_dir(), log_config)

    return system_manager, controller, training_manager, visualizer, plot_kwargs


def net_train_alg1(system_manager:SystemManager|list[SystemManager], controller:Controller, training_manager:TrainingManager, visualizer:Visualizer,
                   **plot_kwargs):

    print("Training!")
    eval_traj_sets, training_flawed = training_manager.controller_training_1(system_manager, controller)
    if not training_flawed:
        print("Training Done!")
    else:
        print("Error occurred during training :'(")
    
    controller_save_dir = ospath_join(get_save_dir(), "model")
    makedirs(controller_save_dir, exist_ok=True)
    controller.save_model(controller_save_dir)
    print("Model Saved!")
    
    if "file_name" not in plot_kwargs.keys():
        plot_kwargs["file_name"] = str(controller)
    if "fig_name" not in plot_kwargs.keys():
        plot_kwargs["fig_name"] = plot_kwargs["file_name"]
    
    tr_kwargs = plot_kwargs.copy()
    
    print("Plotting evaluation trajectories!")
    enumeration_start = 1
    for i, set in enumerate(eval_traj_sets, start=enumeration_start):
        nr = str(training_manager.plot_rate * (i - enumeration_start) + enumeration_start)
        if i == len(eval_traj_sets) - 1 + enumeration_start: nr = str(len(eval_traj_sets) - 1 + enumeration_start)
        if i < 10:
            nr = "0" + nr
        set_name = "Session " + nr
    
        for idx, trex in enumerate([t.export_trajectory() for t in set], start=1):
            if idx < 10: tr_nr = "0" + str(idx)
            else : tr_nr = str(idx)
            tr_kwargs["fig_name"] = plot_kwargs["fig_name"] + " - " + set_name + ", Figure " + tr_nr
            tr_kwargs["file_name"] = plot_kwargs["file_name"] + " " + set_name + " fig" + tr_nr
            visualizer.visualize_state_trajectory(trex, **tr_kwargs)

def run_simulation(system_manager:SystemManager, controller:Controller, config:dict):

    s = system_manager.read_polar_state()
    for i in range(config["simulation_max_length"]):
        
        u = controller.get_control_signal(s)
        system_manager.tick(u)
        s = system_manager.read_polar_state()


if __name__ == "__main__":

    print("Running")

    ##### -- Alfa Simulation:
    #  system_manager, controller, visualizer, config = configuration_alfa()
    #  alfa_simulation(config, system_manager, controller, visualizer)
    ##### -- ## #

    ##### -- Controller alg1
    system_manager, controller, training_manager, visualizer, plot_kwargs = configuration_DDPG_config_1()
    net_train_alg1(system_manager, controller, training_manager, visualizer, **plot_kwargs)
    ##### -- ## #

    #### ### ## #
    print("Finished")