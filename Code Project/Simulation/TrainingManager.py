import tensorflow as tf

from Control.ReplayBuffer import ReplayBuffer
from Control.Rewards import RewardBase
from PhysicalEnvironment.StateTrajectory import TrajectoryExport, Trajectory
from PhysicalEnvironment.SystemManager import SystemManager
from Control.Controller import Controller
from Visualization.Logger import LossLogger, RewardPenaltyLogger
from Visualization.Visualizer import Visualizer



class TrainingManager:

    def __init__(self, **kwargs):
        self.setup(**kwargs)
    
    def setup(self, random_action_process, replay_buffer:ReplayBuffer,
              loss_logger:LossLogger, reward_fn:RewardBase,
              evaluation_kwargs:dict, evaluation_alg:callable,
              rew_pen_logger:RewardPenaltyLogger=None, training_trajs:list[Trajectory]=None,
              visualizer:Visualizer=None,
              n_sessions=1, plot_session_rate=1, repbuffer_reset_keep_fraction=0.0):
        
        self.random_action_process = random_action_process
        self.replay_buffer:ReplayBuffer = replay_buffer
        self.loss_logger = loss_logger
        self.reward_fn:RewardBase = reward_fn
        self.evaluation_kwargs = evaluation_kwargs
        self.evaluation_alg = evaluation_alg
        self.rew_pen_logger = rew_pen_logger
        self.training_trajs = training_trajs
        self.visualizer = visualizer
        self.n_sessions = n_sessions
        self.plot_rate = plot_session_rate
        self.repbuffer_reset_keep_fraction = repbuffer_reset_keep_fraction

    def controller_training_1(self, system_manager: SystemManager | list[SystemManager], controller: Controller,
                              evaluation_sm: None | SystemManager = None):

        plotted_trajs:list[list[Trajectory]] = []

        if evaluation_sm is None:
            if type(system_manager) is list: evaluation_sm = system_manager[0]
            else: evaluation_sm = system_manager

        #controller.config_training(**self.model_training_kwargs)
        training_flawed = False

        for i in range(self.n_sessions):
            print("Training in session " + str(i))

            #controller.train(system_manager, self.random_action_process, self.reward_fn, self.replay_buffer, self.loss_logger,
            #                 rew_pen_logger=self.rew_pen_logger, training_trajlog=self.training_trajs)

            try:
                controller.train(system_manager, self.random_action_process, self.reward_fn, self.replay_buffer, self.loss_logger,
                                 rew_pen_logger=self.rew_pen_logger, training_trajlog=self.training_trajs)
            except Exception as e:
                if 'NaN' in e.message:
                    print("NaN occurred during training. Saving training trajectories and logs. Exiting training loop. Error message:\n")
                    print(e.message)
                    if self.rew_pen_logger is not None:
                        self.rew_pen_logger.save_rew_pen_plots(name_suffix="sess_" + str(i))
                    self.loss_logger.save_loss_plots(name_suffix="sess_" + str(i))
                    if self.training_trajs is not None:
                        for idx, traj in enumerate(self.training_trajs):
                            if traj.point_count < 3:
                                continue
                            train_traj_file_name = "Training_Session_" + str(i) + "_ep_" + str(idx)
                            self.visualizer.visualize_state_trajectory(traj.export_trajectory(), file_name=train_traj_file_name, fig_name=train_traj_file_name)
                    training_flawed = True
                    return plotted_trajs, training_flawed
                else:
                    raise e
            
            self.replay_buffer.reset(keep_fraction=self.repbuffer_reset_keep_fraction)
            
            if self.training_trajs is not None:
                for idx, traj in enumerate(self.training_trajs):
                    train_traj_file_name = "Training_Session_" + str(i) + "_ep_" + str(idx)
                    veh_shows = None
                    if traj.point_count < self.visualizer.vehicle_show_count:
                        print("Warning! Training trajectory had a point count of " + str(traj.point_count))
                        veh_shows = traj.point_count
                    self.visualizer.visualize_state_trajectory(traj.export_trajectory(), file_name=train_traj_file_name, fig_name=train_traj_file_name, n_vehicle_markers=veh_shows)
                    traj.reset()
            
            if not i % self.plot_rate or i == self.n_sessions - 1:
                print("Evaluation in session " + str(i))
                trajs = self.evaluation_alg(evaluation_sm, controller, self.reward_fn, **self.evaluation_kwargs)
                plotted_trajs.append(trajs)
            
            if self.rew_pen_logger is not None:
                self.rew_pen_logger.save_rew_pen_plots(name_suffix="sess_" + str(i))
                self.rew_pen_logger.reset()
            self.loss_logger.save_loss_plots(name_suffix="sess_" + str(i))
            self.loss_logger.reset()
        
        return plotted_trajs, training_flawed