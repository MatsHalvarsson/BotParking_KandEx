#import statements
from PhysicalEnvironment.StateInitializers import StateInitializer
from .SystemState import SystemState
from .SystemDynamics import SystemDynamics
from .StateTrajectory import Trajectory
import numpy as np
from tensorflow import Tensor as tfTensor, math as tfMath, constant as tfConstant, Variable as tfVariable
from tensorflow import convert_to_tensor, stack
from tensorflow import float32 as tfFloat32, bool as tfBool




class SystemManager:
    
    """
    Manages the physical environment, in,
        how it evolves over time,
        state space representations and how they update,
        ## - holds a model of the system dynamics, # - isn't this a controller part? or no...

    Class variables:

        _environment_state := [SystemState] the current state of the system.
        _dynamics_model := [SystemDynamics] the chosen implementation of system dynamics to be used during run.
        __dt := [float] the length of one time step, should remain constant after initialization.

    Class methods:

        __init__ ():
            initializes all class variables

        reset_environment (e, a, d):
            resets/initiates environment to a given configuration.

        tick (u, w):    |    control inputs u, w || u := forwards linear velocity | w := angular velocity
            take one step forwards in time,
            commands the computations for updating the system.
    
        read_state (cartesian_form=false):
            returns the current state of the system variables,
            caller can choose to receive a cartesian representation but polar rep should be default.
    
    """

    def __init__(self, time_step_length, epsilon, goal_function:callable,
                 out_of_bounds_function:callable,
                 state_initializer: StateInitializer=None,
                 state_class=SystemState, dynamics_class=SystemDynamics):
        self._environment_state = state_class(state_initializer)
        self._dynamics_model = dynamics_class(epsilon)
        self.__dt:tfTensor = tfConstant(time_step_length, dtype=tfFloat32)
        self._current_time = tfVariable(0, dtype=tfFloat32, trainable=False)
        self._epsilon = tfConstant(epsilon, dtype=tfFloat32)
        #self._increased_resolution = tfConstant(False, dtype=tfBool)
        self._goal_function = goal_function
        self._oob_fn = out_of_bounds_function

    @property
    def time(self):
        return self._current_time.read_value()
        
    def reset_environment(self):
        self._environment_state.initialize_state()
        self._current_time.assign(0)
    
    def tick(self, control_vars: tfTensor) -> tuple[tfTensor, bool, bool]:
        #the if-statement below does not work in tf.function I think, because "not tensor" evaluates to a python bool
        #if not self._increased_resolution and self.read_polar_state()[0] + self._epsilon < 2 * self.__dt:
        #    self.__dt = self.__dt / 10
        #    self._increased_resolution = tfConstant(True, dtype=tfBool)
        incs = self._dynamics_model.calculate_state_increments(self.__dt, self._environment_state.current_state, control_vars)
        self._environment_state.update_state_variables(incs)
        self._current_time.assign_add(self.__dt, read_value=False)
        #if self._increased_resolution and self.read_polar_state()[0] + self._epsilon > 20 * self.__dt:
        #    self.__dt = self.__dt * 10
        #    self._increased_resolution = tfConstant(False, dtype=tfBool)
        return incs, self.is_goal(), self.is_out_of_bounds()
    
    def read_polar_state(self) -> tfTensor:
        return self._environment_state.current_state
    
    def read_cartesian_state(self) -> tfTensor:
        return self._environment_state.current_state_cartesian
    
    def is_goal(self) -> bool:
        return self._goal_function(self.read_polar_state())
    
    def is_out_of_bounds(self) -> bool:
        return self._oob_fn(self.read_polar_state())


class BatchedSystemManager:
    def __init__(self, n_systems=8, *sm_args, **sm_kwargs):
        self._sm_list:list[SystemManager] = []*n_systems
        for i in range(n_systems):
            self._sm_list[i] = SystemManager(*sm_args, **sm_kwargs)
    
    def read_states(self, polar_rep=True) -> tfTensor:
        if polar_rep:
            return stack([sm.read_polar_state() for sm in self._sm_list], axis=0)
        return stack([sm.read_cartesian_state() for sm in self._sm_list], axis=0)
    
    def tick(self, c_var_list:list[tfTensor]) -> list[bool]:
        return [sm.tick(c) for sm, c in zip(self._sm_list, c_var_list)]
    
    #def tick_on_unfinished
    
    def reset_environment(self):
        for sm in self._sm_list:
            sm.reset_environment()
    
    def is_goal(self) -> list[bool]:
        return [sm.is_goal() for sm in self._sm_list]