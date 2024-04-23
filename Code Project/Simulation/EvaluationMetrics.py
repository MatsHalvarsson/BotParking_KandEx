import tensorflow as tf
from numpy import pi

from ..PhysicalEnvironment.StateTrajectory import TrajectoryExport as TrEx




def metric_alfa(trajectories:list[TrEx], num_trs=None):
    if not num_trs:
        num_trs = len(trajectories)
    
    metrics:list[dict] = []
    for trex in trajectories:
        r_tot = tf.reduce_sum(trex.rewards)
        orientation_difficulty = tf.sqrt(2 * tf.square(trex.polar_states[0,1] / pi + 1.0 / 6.0) + 3 * tf.square(trex.polar_states[0,2] / pi))
        initial_difficulty = orientation_difficulty * tf.sqrt(trex.polar_states[0,0])

        d = {
            "Total Reward" : r_tot,
            "Initial Difficulty" : initial_difficulty,
            "Orientation Difficulty" : orientation_difficulty,
            "General Score" : r_tot / initial_difficulty,
            "Distance Score" : r_tot / trex.polar_states[0,0],
            "sum u" : tf.reduce_sum(trex.control_inputs[:,0]),
            "sum u^2" : tf.reduce_sum(tf.square(trex.control_inputs[:,0])),
            "sum w" : tf.reduce_sum(trex.control_inputs[:,1]),
            "sum w^2" : tf.reduce_sum(tf.square(trex.control_inputs[:,1]))
        }
        metrics.append(d)
    return metrics

def metric_fn1(trajectories:list[TrEx], num_trs=None): #not finished
    if not num_trs:
        num_trs = len(trajectories)
    
    metrics = []
    for trex in trajectories:
        r_tot = tf.reduce_sum(trex.rewards)
        #orientation_difficulty = 2.0 / pi * tf.square(trex.polar_states[0,1]) + 3.0 / pi * tf.square(trex.polar_states[0,2])
        orientation_difficulty = tf.sqrt(2 * tf.square(trex.polar_states[0,1] / pi + 1.0 / 6.0) + 3 * tf.square(trex.polar_states[0,2] / pi))
        initial_difficulty = orientation_difficulty * tf.sqrt(trex.polar_states[0,0])

        #bad metrics because rewards are logged in each point independent of the distance between each point and where on the path the
        #point count is locally dense and where it is sparse
        d = {"Total Reward" : r_tot,
             "Reward per point" : r_tot / trex.point_count,
             "General Score" : r_tot / initial_difficulty,
             "Distance Score" : r_tot / trex.polar_states[0,0],
             "Orientation Difficulty" : orientation_difficulty,
             "Initial Difficulty" : initial_difficulty}
        metrics.append(d)
