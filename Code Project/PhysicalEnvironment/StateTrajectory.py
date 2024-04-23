#import statements
import tensorflow as tf
import numpy as np

from PhysicalEnvironment.SystemState import SystemState



class TrajectoryExport:
    def __init__(self, t_dict):
        self._t_dict = t_dict
        self.__return_as_np = False
        self._np_dict = None
    
    staticmethod
    def trajectory_in_TrEx_format(trajectory_stack, length:tf.Tensor, point_count, reached_goal,
                                  n_states=3, n_control_vars=2):
        trajectory_stack = trajectory_stack
        d = {"polar_states" : trajectory_stack[:, :n_states],
             "control_inputs" : trajectory_stack[:, n_states : n_states + n_control_vars],
             "rewards" : trajectory_stack[:, -2],
             "time" : trajectory_stack[:, -1],
             "length" : length,
             "point_count" : point_count,
             "reached_goal" : reached_goal}
        d["cartesian_states"] = SystemState.convert_to_cartesian(d["polar_states"])
        return d
    
    @property
    def return_as_np(self):
        return self.__return_as_np
    
    @return_as_np.setter
    def return_as_np(self, v):
        self.__return_as_np = v
        if not self._np_dict and v:
            self._np_dict = {"polar_states" : self._t_dict["polar_states"].numpy(),
                             "cartesian_states" : self._t_dict["cartesian_states"].numpy(),
                             "control_inputs" : self._t_dict["control_inputs"].numpy(),
                             "rewards" : self._t_dict["rewards"].numpy(),
                             "time" : self._t_dict["time"].numpy(),
                             "length" : self._t_dict["length"].numpy(),
                             "point_count" : self._t_dict["point_count"].numpy(),
                             "reached_goal" : self._t_dict["reached_goal"].numpy()}

    @property
    def polar_states(self):
        if self.return_as_np:
            return self._np_dict["polar_states"]
        return self._t_dict["polar_states"]
    
    @property
    def cartesian_states(self):
        if self.return_as_np:
            return self._np_dict["cartesian_states"]
        return self._t_dict["cartesian_states"]
    
    @property
    def control_inputs(self):
        if self.return_as_np:
            return self._np_dict["control_inputs"]
        return self._t_dict["control_inputs"]
    
    @property
    def rewards(self):
        if self.return_as_np:
            return self._np_dict["rewards"]
        return self._t_dict["rewards"]
    
    @property
    def time(self):
        if self.return_as_np:
            return self._np_dict["time"]
        return self._t_dict["time"]
    
    @property
    def length(self):
        if self.return_as_np:
            return self._np_dict["length"]
        return self._t_dict["length"]
    
    @property
    def point_count(self):
        if self.return_as_np:
            return self._np_dict["point_count"]
        return self._t_dict["point_count"]
    
    @property
    def reached_goal(self):
        if self.return_as_np:
            return self._np_dict["reached_goal"]
        return self._t_dict["reached_goal"]



class Trajectory:

    def __init__(self, expected_total_points=100, num_state_vars=3, num_control_vars=2):
        self._n = tf.constant(0, dtype=tf.int32)
        self._nums_s = tf.convert_to_tensor(num_state_vars)
        self._nums_c = tf.convert_to_tensor(num_control_vars)
        self._tArray = tf.TensorArray(dtype=tf.float32, size=expected_total_points, element_shape=(num_state_vars+num_control_vars+2,), dynamic_size=True, clear_after_read=False)
        self._reached_goal = tf.Variable(False, dtype=tf.bool, shape=())
        self.length = tf.Variable(0, dtype=tf.float32, shape=())
    
    @property
    def point_count(self):
        return self._n

    @property
    def states(self):
        return self._tArray.stack()[0:self._n, :self._nums_s]
    
    @property
    def control_inputs(self):
        return self._tArray.stack()[0:self._n, self._nums_s:self._nums_s+self._nums_c]
    
    @property
    def rewards(self):
        return self._tArray.stack()[0:self._n, -2]
    
    @property
    def time_steps(self):
        return self._tArray.stack()[0:self._n, -1]
    
    @property
    def reached_goal(self):
        return self._reached_goal.read_value()
    
    @reached_goal.setter
    def reached_goal(self, v):
        self._reached_goal.assign(v)
    
    def reset(self):
        size = self._tArray.size()
        self._tArray.close()
        self._n = tf.constant(0, dtype=tf.int32)
        self._tArray = tf.TensorArray(dtype=tf.float32, size=size, element_shape=(self._nums_s+self._nums_c+2,), dynamic_size=True, clear_after_read=False)
        self._reached_goal.assign(False, read_value=False)
        self.length.assign(0.0, read_value=False)
    
    def add_point(self, state, control_inputs, reward, time_step):
        self._tArray = self._tArray.write(self._n, tf.concat([state, control_inputs, tf.reshape([reward, time_step],[-1])], axis=0))
        self._n += 1
        if self._n > 1:
            s = self._tArray.read(self._n-2)
            dist = tf.square(s[0]) + tf.square(state[0]) - 2 * s[0] * state[0] * tf.cos(s[2] - state[2])
            if dist > 0.0:
                self.length.assign_add(tf.sqrt(dist))
    
    def get_trajectory_as_dict(self): #not used
        stack = self._tArray.stack()
        return {"polar_states" : stack[0:self._n, :self._nums_s],
              "control_inputs" : stack[0:self._n, self._nums_s:self._nums_s+self._nums_c],
              "rewards" : stack[0:self._n, -2],
              "time" : stack[0:self._n, -1],
              "length" : self.length.read_value(),
              "point_count" : self.point_count,
              "reached_goal" : self._reached_goal.read_value()
        }
    
    def export_trajectory(self):
        #d = self.get_trajectory_as_dict()
        #d["cartesian_states"] = SystemState.convert_to_cartesian(d["polar_states"])
        d = TrajectoryExport.trajectory_in_TrEx_format(self._tArray.stack()[:self._n,:], self.length.read_value(),
                                                       self.point_count, self.reached_goal,
                                                       n_states=self._nums_s, n_control_vars=self._nums_c)
        return TrajectoryExport(d)


class NormalizedTrajectory(Trajectory):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm_vector = tf.constant([1.0, np.pi, np.pi], dtype=tf.float32, shape=(3,))

    def add_point(self, state, control_inputs, reward, time_step):
        return super().add_point(tf.multiply(state, self.norm_vector), control_inputs, reward, time_step)