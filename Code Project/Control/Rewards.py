from tensorflow import math as tfMath, float32 as tfFloat32, int32 as tfInt32, constant as tfConstant, get_static_value, cast, stack
from tensorflow import square, reduce_sum as tfSum, maximum as tfMax, minimum as tfMin
from numpy import pi



class RewardBase:
    def __init__(self):
        pass

    def __call__(self, old_state=None, action=None, new_state=None, state_incs=None, is_goal=False, is_oob=False):
        """Only use arguments with keywords. Any positional arguments are likely to cause errors. This was a stylistic choice."""
        print("Warning! RewardBase class called for reward! Should not be intended!")
        return NoReward.reward()
    
    def __str__(self):
        return "Reward Base Class"
    
    def reset(self, s0):
        pass

class RewardPlusPenalizer(RewardBase):
    def __init__(self, rewarder, penalizer, r_weight=1.0, p_weight=1.0):
        self.rewarder = rewarder
        self.penalizer = penalizer
        #self.relative_weights = tfConstant([r_weight, p_weight], dtype=tfFloat32, shape=(2,))
        self.r_scale = tfConstant(r_weight, dtype=tfFloat32)
        self.p_scale = tfConstant(p_weight, dtype=tfFloat32)

        str_r = str(r_weight)
        str_r = str_r[0 : min(len(str_r), 3)]
        if str_r[-1] == ".": str_r = str_r[:-1]
        
        str_p = str(p_weight)
        str_p = str_p[0 : min(len(str_p), 3)]
        if str_p[-1] == ".": str_p = str_p[:-1]

        self.end_str = ", wt-" + str_r + ":" + str_p
        ##
    
    def get_penalty_evaluation(self, **kwargs):
        return self.penalizer(**kwargs)
    
    def get_reward_evaluation(self, **kwargs):
        return self.rewarder(**kwargs)
    
    def __call__(self, **kwargs):
        return self.r_scale * self.rewarder(**kwargs) + self.p_scale * self.penalizer(**kwargs)
    
    def __str__(self):
        return str(self.rewarder) + ", " + str(self.penalizer) + self.end_str
    
    def reset(self, s0):
        self.rewarder.reset(s0)
        self.penalizer.reset(s0)

class NormalRewardOrGoalRandOOBR(RewardBase):
    def __init__(self, normal_R_obj, goal_R=100.0, oob_R=-100.0):
        self.normal_R = normal_R_obj
        #self.relative_weights = tfConstant([r_weight, p_weight], dtype=tfFloat32, shape=(2,))
        self.goal_R = tfConstant(goal_R, dtype=tfFloat32)
        self.oob_R = tfConstant(oob_R, dtype=tfFloat32)

        str_gR = str(goal_R).split(".")[0]
        if str_gR[-1] == ".": str_gR = str_gR[:-1]
        
        str_oobR = str(oob_R).split(".")[0]
        if str_oobR[-1] == ".": str_oobR = str_oobR[:-1]

        self.end_str = ", Or, goal: " + str_gR + ", OOB: " + str_oobR
    
    def __call__(self, is_goal=False, is_oob=False, **kwargs):
        if is_goal: return self.goal_R
        if is_oob: return self.oob_R
        return self.normal_R(**kwargs)
    
    def __str__(self):
        return str(self.normal_R) + self.end_str

class TanhAppliedReward(RewardBase):
    def __init__(self, underlying_reward_obj:RewardBase, scale=1.0, threshold_reward=10.0, threshold_value=0.9):
        self.underlying_rewarder = underlying_reward_obj
        self.scale = tfConstant(scale, dtype=tfFloat32, shape=())
        self.weight = tfConstant(tfMath.log((1 + threshold_value) / (1 - threshold_value)) / (2 * threshold_reward),
                                 dtype=tfFloat32, shape=())
        self.thresh = threshold_reward
        self.scale_str = str(scale)[:min(len(str(scale)), 3)]
        if self.scale_str[-1] == ".": self.scale_str = self.scale_str[:-1]
    
    def __call__(self, **kwargs):
        rew = self.underlying_rewarder(**kwargs)
        if tfMath.abs(rew) > self.thresh: return self.scale * tfMath.sign(rew)
        rew = rew * self.weight
        e_to_x = tfMath.exp(rew)
        e_to_negx = tfMath.exp(- rew)
        return tfMath.multiply(self.scale, tfMath.divide(e_to_x - e_to_negx, e_to_x + e_to_negx))
    
    def __str__(self):
        return "tanh scale:{s}, ".format(s=self.scale_str) + str(self.underlying_rewarder)
    
    def reset(self, s0):
        return self.underlying_rewarder.reset(s0)

class RewardAlfa(RewardBase):
    def reward(self, new_state):
        return - tfMath.reduce_sum(tfMath.abs(new_state))
    
    #@staticmethod
    #def reward(new_state):
    #    return - tfMath.reduce_sum(tfMath.abs(new_state))

    #@staticmethod
    #def call(old_state, action, new_state):
    #    return RewardAlfa.reward(new_state)

    def __call__(self, new_state=None, **kwargs):
        return self.reward(new_state)
    
    def __str__(self):
        return "Reward Function Alfa"

class NoReward(RewardBase):
    @staticmethod
    def reward(**kwargs):
        return tfConstant(0.0, dtype=tfFloat32, shape=())

    def __call__(self, **kwargs):
        return self.reward()
    
    def __str__(self):
        return "No Reward"

class NormalizedLyapunovReward(RewardBase):
    def __init__(self, lamb=1.0, h=1.0):
        self.consts = tfConstant([lamb, 1.0, h], dtype=tfFloat32, shape=(3,))
        self._s0_LypunvEval = tfConstant(1.0, dtype=tfFloat32, shape=())

    def __call__(self, new_state=None, **kwargs):
        return - tfSum(tfMath.multiply(self.consts, square(new_state))) / 2.0 / self._s0_LypunvEval

    def reset(self, s0):
        self._s0_LypunvEval = tfSum(tfMath.multiply(self.consts, square(s0))) / 2.0
    
    def __str__(self):
        return "s_0 norm Lya-R lamb: {lamb}, $h$: {h}".format(lamb=self.consts[0], h=self.consts[2])

class ComparativeLyapunovReward(RewardBase):
    def __init__(self, dt, lamb=1.0, h=1.0):
        self.str_lamb = str(lamb)[0:min(len(str(lamb)), 3)]
        if self.str_lamb[-1] == ".": self.str_lamb = self.str_lamb[0:-1]
        self.str_h = str(h)[0:min(len(str(h)), 3)]
        if self.str_h[-1] == ".": self.str_h = self.str_h[0:-1]
        self.consts = tfConstant([lamb, 1.0, h], dtype=tfFloat32, shape=(3,))
        self.dt = tfConstant(dt, dtype=tfFloat32, shape=())

    def __call__(self, old_state=None, new_state=None, **kwargs):
        return tfSum(tfMath.multiply(self.consts, square(old_state) - square(new_state))) / 2.0
    
    def __str__(self):
        return "Comp Lya-R lamb: {lamb}, $h$: {h}".format(lamb=self.str_lamb, h=self.str_h)

class ComparativeLocallyNormalizedLyapunovReward(RewardBase):
    def __init__(self, dt, lamb=1.0, h=1.0, epsilon=1e-4):
        self.str_lamb = str(lamb)[0:min(len(str(lamb)), 3)]
        if self.str_lamb[-1] == ".": self.str_lamb = self.str_lamb[0:-1]
        self.str_h = str(h)[0:min(len(str(h)), 3)]
        if self.str_h[-1] == ".": self.str_h = self.str_h[0:-1]
        self.consts = tfConstant([lamb, 1.0, h], dtype=tfFloat32, shape=(3,))
        self.dt = tfConstant(dt, dtype=tfFloat32, shape=())
        self.epsilon = tfConstant(epsilon, dtype=tfFloat32, shape=())

    def __call__(self, old_state=None, new_state=None, **kwargs):
        return tfMath.divide(tfSum(tfMath.multiply(self.consts, square(old_state) - square(new_state))),
                             self.epsilon + tfSum(tfMath.multiply(self.consts, square(old_state)))) / self.dt
    
    def __str__(self):
        return "Comp LclNorm Lya-R lamb: {lamb}, $h$: {h}".format(lamb=self.str_lamb, h=self.str_h)

#class LogComparativeLyapunovReward(RewardBase):
#    def __init__(self, lamb=1.0, h=1.0):
#        self.consts = tfConstant([lamb, 1.0, h], dtype=tfFloat32, shape=(3,))
#
#    def __call__(self, old_state, action, new_state):
#        return tfSum(tfMath.multiply(self.consts, square(old_state)) - tfMath.multiply(self.consts, square(new_state))) / 2.0
#    
#    def __str__(self):
#        return "Comparative Lyapunov Reward, lamb: {lamb}, $h$: {h}".format(lamb=self.consts[0], h=self.consts[2])

class NegativeLyapunovDerivative(RewardBase):

    def __init__(self, dt, lamb=1.0, h=1.0):
        self.str_lamb = str(lamb)[0:min(len(str(lamb)), 3)]
        if self.str_lamb[-1] == ".": self.str_lamb = self.str_lamb[0:-1]
        self.str_h = str(h)[0:min(len(str(h)), 3)]
        if self.str_h[-1] == ".": self.str_h = self.str_h[0:-1]
        self.consts = tfConstant([lamb, 1.0, h], dtype=tfFloat32, shape=(3,))
        self.dt = tfConstant(dt, dtype=tfFloat32, shape=())
    
    def __call__(self, old_state=None, state_incs=None, **kwargs):
        return - tfMath.reduce_sum(tfMath.multiply(self.consts, tfMath.multiply(old_state, state_incs))) / self.dt
    
    def __str__(self):
        return "Neg. Lyap Deriv lamb: {lamb}, $h$: {h}".format(lamb=self.str_lamb, h=self.str_h)

class PositiveLyapunovDerivative(RewardBase):

    def __init__(self, dt, lamb=1.0, h=1.0):
        self.str_lamb = str(lamb)[0:min(len(str(lamb)), 3)]
        if self.str_lamb[-1] == ".": self.str_lamb = self.str_lamb[0:-1]
        self.str_h = str(h)[0:min(len(str(h)), 3)]
        if self.str_h[-1] == ".": self.str_h = self.str_h[0:-1]
        self.consts = tfConstant([lamb, 1.0, h], dtype=tfFloat32, shape=(3,))
        self.dt = tfConstant(dt, dtype=tfFloat32, shape=())
    
    def __call__(self, old_state=None, state_incs=None, **kwargs):
        return tfMath.reduce_sum(tfMath.multiply(self.consts, tfMath.multiply(old_state, state_incs))) / self.dt
    
    def __str__(self):
        return "Pos. Lyap Deriv lamb: {lamb}, $h$: {h}".format(lamb=self.str_lamb, h=self.str_h)

class IndividualAbsLyapunovDerivative(RewardBase):

    def __init__(self, dt, lamb=1.0, h=1.0):
        self.str_lamb = str(lamb)[0:min(len(str(lamb)), 3)]
        if self.str_lamb[-1] == ".": self.str_lamb = self.str_lamb[0:-1]
        self.str_h = str(h)[0:min(len(str(h)), 3)]
        if self.str_h[-1] == ".": self.str_h = self.str_h[0:-1]
        self.consts = tfConstant([lamb, 1.0, h], dtype=tfFloat32, shape=(3,))
        self.dt = tfConstant(dt, dtype=tfFloat32, shape=())
    
    def __call__(self, old_state=None, state_incs=None, **kwargs):
        return - tfMath.reduce_sum(tfMath.multiply(self.consts, tfMath.abs(tfMath.multiply(old_state, state_incs)))) / self.dt
    
    def __str__(self):
        return "indiv. abs Lyap Deriv lamb: {lamb}, $h$: {h}".format(lamb=self.str_lamb, h=self.str_h)

class LyapDeriv_eq11(RewardBase):

    def __init__(self, lambdagamma=1.0, k=1.0):
        self.str_lambdagamma = str(lambdagamma)[0:min(len(str(lambdagamma)), 3)]
        if self.str_lambdagamma[-1] == ".": self.str_lambdagamma = self.str_lambdagamma[0:-1]
        self.str_k = str(k)[0:min(len(str(k)), 3)]
        if self.str_k[-1] == ".": self.str_k = self.str_k[0:-1]
        self.consts = tfConstant([lambdagamma, k], dtype=tfFloat32, shape=(2,))
    
    def __call__(self, old_state=None, **kwargs):
        return - tfMath.reduce_sum(
                 tfMath.multiply(self.consts,
                                 tfMath.square(stack([tfMath.cos(old_state[1]) * old_state[0],
                                                      old_state[1]]))))
    
    def __str__(self):
        return "paper Eq11 l*g: {lambdagamma}, $k$: {k}".format(lambdagamma=self.str_lambdagamma, k=self.str_k)

class LyapDeriv_eq11_and_thetaterm(RewardBase):

    def __init__(self, lambdagamma=1.0, k=1.0, k2=1.0):
        self.str_lambdagamma = str(lambdagamma)[0:min(len(str(lambdagamma)), 3)]
        if self.str_lambdagamma[-1] == ".": self.str_lambdagamma = self.str_lambdagamma[0:-1]
        self.str_k = str(k)[0:min(len(str(k)), 3)]
        if self.str_k[-1] == ".": self.str_k = self.str_k[0:-1]
        self.str_k2 = str(k2)[0:min(len(str(k2)), 3)]
        if self.str_k2[-1] == ".": self.str_k2 = self.str_k2[0:-1]
        self.consts = tfConstant([lambdagamma, k, k2], dtype=tfFloat32, shape=(3,))
    
    def __call__(self, new_state=None, **kwargs):
        return - tfMath.reduce_sum(
                 tfMath.multiply(self.consts,
                                 tfMath.square(stack([tfMath.cos(new_state[1]) * new_state[0],
                                                      new_state[1],
                                                      new_state[2]]))))
    
    def __str__(self):
        return "paper Eq11 with theta-term l*g: {lambdagamma}, $k$: {k}, $k2$: {k2}".format(lambdagamma=self.str_lambdagamma, k=self.str_k, k2=self.str_k2)

class LyapDeriv_eq11_comparison(RewardBase):

    def __init__(self, lamb=1.0, k=1.0):
        self.str_lamb = str(lamb)[0:min(len(str(lamb)), 3)]
        if self.str_lamb[-1] == ".": self.str_lamb = self.str_lamb[0:-1]
        self.str_k = str(k)[0:min(len(str(k)), 3)]
        if self.str_k[-1] == ".": self.str_k = self.str_k[0:-1]
        self.consts = tfConstant([lamb, k], dtype=tfFloat32, shape=(2,))
    
    def __call__(self, old_state=None, **kwargs):
        return - tfMath.reduce_sum(tfMath.multiply(tfMath.square(old_state[:2]), self.consts))
    
    def __str__(self):
        return "comparison Eq11 lambda: {lamb}, $k$: {k}".format(lamb=self.str_lamb, k=self.str_k)

class SparseRewarder(RewardBase):
    def __init__(self):
        pass

    def __call__(self, **kwargs):
        pass

    def __str__(self):
        pass

    def reset():
        pass

class ControlSignalPenalizerBase(RewardBase):
    def __init__(self, e0, ang_max, dt, u_max=3.0, w_max=3.0, L_mid=1.0, L_upper_bound=1.5, L_lower_bound=0.0, cost_scaling=1.0):
        self.e0 = tfConstant(e0, dtype=tfFloat32, shape=())
        self.ang_max = tfConstant(ang_max, dtype=tfFloat32, shape=())
        self.dt = tfConstant(dt, dtype=tfFloat32, shape=())
        self.u_max = tfConstant(u_max, dtype=tfFloat32, shape=())
        self.w_max = tfConstant(w_max, dtype=tfFloat32, shape=())
        self.L_mid = tfConstant(L_mid, dtype=tfFloat32, shape=())
        self.L_upper_bound = tfConstant(L_upper_bound, dtype=tfFloat32, shape=())
        self.L_lower_bound = tfConstant(L_lower_bound, dtype=tfFloat32, shape=())
        self.cost_scaling = tfConstant(cost_scaling, dtype=tfFloat32, shape=())

        self.lu = tfConstant(L_mid / u_max, dtype=tfFloat32, shape=())
        self.lw = tfConstant(L_mid / w_max, dtype=tfFloat32, shape=())
        self.l_squares = stack([square(self.lu), square(self.lw)])

        L_str = str(L_mid)[0:min(len(str(L_mid)),3)]
        if L_str[-1] == "." : L_str = L_str[:-1]
        u_str = str(u_max)[0:min(len(str(u_max)),3)]
        if u_str[-1] == "." : u_str = u_str[:-1]
        w_str = str(w_max)[0:min(len(str(w_max)),3)]
        if w_str[-1] == "." : w_str = w_str[:-1]
        self.L_str = L_str; self.u_str = u_str; self.w_str = w_str
    
    @property
    def p1(self):
        """number of updates for traversing e0 distance with u_max"""
        return  self.e0 / (self.dt * self.u_max)
    
    @property
    def p2(self):
        """number of updates for rotating ang_max with w_max (rotating half a circle)"""
        return tfMath.sqrt(2.0) * self.ang_max / (self.dt * self.w_max)
    
    def penalty_on_Lupp_plus_Lmid(self):
        return self.penalty(self.signal_Lcost_transgression(self.L_mid + self.L_upper_bound))
    
    def penalty_on_Zero_Signal(self):
        return self.penalty(self.signal_Lcost_transgression(0.0))
    
    def signal_Lcost_transgression(self, signal):
        return tfMath.add(tfMax(signal - self.L_upper_bound, 0.0), tfMax(self.L_lower_bound - signal, 0.0))
    
    def signal_L(self, signal):
        return tfMath.sqrt(tfSum(tfMath.multiply(square(signal), self.l_squares)))
    
    def penalty(self, signal_transgression): #overload this in subclasses
        return self.cost_scaling * signal_transgression
    
    def __call__(self, action=None, **kwargs):
        return - self.penalty(self.signal_Lcost_transgression(self.signal_L(action)))
    
    def __str__(self): #overload this in subclasses to add info
        return "Pen: (L, u, w)=({L}, {u}, {w})".format(L=self.L_str, u=self.u_str, w=self.w_str)

class QuadraticPenalizer(ControlSignalPenalizerBase):
    def penalty(self, signal_transgression):
        return square(self.cost_scaling * signal_transgression)
    
    def __str__(self):
        return "Sqr " + super().__str__()

class LogarithmicPenalizer(ControlSignalPenalizerBase):
    def penalty(self, signal_transgression):
        return tfMath.log(self.cost_scaling * signal_transgression + 1.0)
    
    def __str__(self):
        return "Log " + super().__str__()