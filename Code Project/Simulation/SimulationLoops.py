import tensorflow as tf
from Control.Controller import Controller

from Control.MAicardiController import MAicardiController
from Control.Rewards import RewardBase
from PhysicalEnvironment.StateTrajectory import Trajectory
from PhysicalEnvironment.SystemManager import SystemManager
from Visualization.Visualizer import Visualizer





def alfa_simulation(config, system_manager: SystemManager, controller: MAicardiController, visualizer: Visualizer):

    #This function is not updated with goal reward or out of bounds penalty... Look through the code before using again

    trajectories:list[Trajectory] = []

    for ep in range(config["episode_count"]):
        system_manager.reset_environment()
        trajectories.append(Trajectory())
        s = system_manager.read_polar_state()
        T = config["episode_max_length"]
        for _ in range(T):
            c = controller.get_control_signal(s)
            t = system_manager.time
            s_incs, _, _ = system_manager.tick(c)
            r = config["reward_function"](old_state=s, action=c, new_state=system_manager.read_polar_state(), state_incs=s_incs)
            trajectories[-1].add_point(s, c, r, t)
            s = system_manager.read_polar_state()
            if system_manager.is_goal():
                trajectories[-1].reached_goal = True
                break
        trajectories[-1].add_point(s, tf.constant([0,0], dtype=tf.float32), tf.constant(0, dtype=tf.float32), system_manager.time)

    for idx, trex in enumerate([t.export_trajectory() for t in trajectories]):
        visualizer.visualize_state_trajectory(tr_ex=trex,
                                              controller_info=config["controller_info"], simulation_info=config["simulation_info"],
                                              file_name=config["fig_name"] + "_run_" + str(idx),
                                              fig_name=config["fig_name"] + "_run_" + str(idx))


def controller_test_alg1(sm :SystemManager, controller:Controller, reward_fn:RewardBase,
                         num_eps=5, ep_length=100, trajectory_type=Trajectory):
    
    trajectories:list[trajectory_type] = []

    for ep in range(num_eps):
        sm.reset_environment()
        trajectories.append(trajectory_type())
        s = sm.read_polar_state()
        
        for t in range(ep_length):
            c = controller.get_control_signal(s)
            t = sm.time
            s_incs, is_goal, is_oob = sm.tick(c)
            snew = sm.read_polar_state()
            r = reward_fn(old_state=s, action=c, new_state=snew, state_incs=s_incs, is_goal=is_goal, is_oob=is_oob)
            trajectories[-1].add_point(s,c,r,t)
            s = snew
            if is_goal:
                trajectories[-1].reached_goal = True
                break
            if is_oob:
                break
        trajectories[-1].add_point(s, tf.constant([0,0], dtype=tf.float32), tf.constant(0.0, dtype=tf.float32), sm.time)
    
    return trajectories