from tensorflow import Tensor as tfTensor
from Control.ReplayBuffer import ReplayBuffer
from Control.Rewards import RewardBase
from PhysicalEnvironment.SystemManager import SystemManager




class Controller:
    """
    Base class for controller implementations.
    
    """

    def reinitialize(self):
        print("Controller base class training not meant to be defined.")
    
    def config_training(self, **kwargs):
        print("Controller base class training not meant to be defined.")

    def train(self, sm:SystemManager, random_action_process:callable, reward_func:RewardBase, replay_buffer:ReplayBuffer):
        print("Controller base class training not meant to be defined.")
    
    def get_control_signal(self, environment_state: tfTensor):
        print("Controller base class output not meant to be defined.")
    
    def __str__(self):
        return str(type(self))
    
    def save_model(self, save_dir):
        print("Controller base class model saving not defined!")