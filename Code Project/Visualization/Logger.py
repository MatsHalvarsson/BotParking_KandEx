from tensorflow import TensorArray as ta, reshape, math as tfMath
from tensorflow import float32 as tfFloat32
import numpy as np
from matplotlib import pyplot as plt
from os.path import join as ospath_join




class LossLogger:

    def __init__(self, save_dir, expected_num_logs=1000, fig_name="Losses", runavg_interval=10):
        self.exp_logs = expected_num_logs
        self.reset()
        self.fig_name = fig_name
        self.save_dir = save_dir
        self.runavg_interval = runavg_interval

    def add_actor_loss(self, loss):
        self._actr_ta = self._actr_ta.write(self._act_i, loss)
        self._act_i += 1
    
    def add_critic_loss(self, loss):
        self._crit_ta = self._crit_ta.write(self._crit_i, loss)
        self._crit_i += 1
    
    def export(self):
        return {"actor" : reshape(self._actr_ta.gather(range(0,self._act_i)), [-1]).numpy(),
                "critic" : reshape(self._crit_ta.gather(range(0,self._crit_i)), [-1]).numpy()}
    
    def export_as_logarithmified(self):
        return {"actor" : tfMath.divide(tfMath.log(1e-7 + tfMath.abs(reshape(self._actr_ta.gather(range(0,self._act_i)), [-1]))), tfMath.log(10.0)).numpy(),
                "critic" : tfMath.divide(tfMath.log(1e-7 + tfMath.abs(reshape(self._crit_ta.gather(range(0,self._crit_i)), [-1]))), tfMath.log(10.0)).numpy()}
    
    def export_as_running_avg(self, interval=10):
        export = self.export()
        act_avg = np.empty_like(export["actor"])
        for i in range(len(act_avg)):
            act_avg[i] = np.sum(export["actor"][max(0, i-interval):i+1]) / (i + 1 - max(0, i-interval))
        crit_avg = np.empty_like(export["critic"])
        for i in range(len(crit_avg)):
            crit_avg[i] = np.sum(export["critic"][max(0, i-interval):i+1]) / (i + 1 - max(0, i-interval+1))
        return {"actor_runavg" : act_avg, "critic_runavg" : crit_avg}
    
    def save_loss_plots(self, name_suffix=""):
        
        fig = plt.figure(layout="constrained", figsize=(10,6), dpi=100)
        fig.suptitle(self.fig_name, fontsize=12)
        axAct = plt.subplot2grid((1,3), (0,0))
        axCri = plt.subplot2grid((1,3), (0,1))
        axLogL = plt.subplot2grid((1,3), (0,2))

        axAct.set_title("Actor")
        axCri.set_title("Critic")
        axLogL.set_title("Log Losses")

        export = self.export()
        avg_export = self.export_as_running_avg(self.runavg_interval)
        log_export = self.export_as_logarithmified()

        axAct.plot(export["actor"], '-r')
        axAct.plot(avg_export["actor_runavg"], '--g')

        axCri.plot(export["critic"], '-r')
        axCri.plot(avg_export["critic_runavg"], '--g')

        axLogL.plot(log_export["actor"], '-r', label="Actor")
        axLogL.plot(log_export["critic"], '-b', label="Critic")
        axLogL.legend()

        if name_suffix != "": name_suffix = "_" + name_suffix

        fig.savefig(ospath_join(self.save_dir, "losses" + name_suffix))
        plt.close(fig)
    
    def reset(self):
        if hasattr(self, "_actr_ta"): self._actr_ta.close()
        if hasattr(self, "_crit_ta"): self._crit_ta.close()
        self._actr_ta = ta(dtype=tfFloat32, size=self.exp_logs,
                      dynamic_size=True, clear_after_read=False,
                      element_shape=())
        self._act_i = 0
        self._crit_ta = ta(dtype=tfFloat32, size=self.exp_logs,
                      dynamic_size=True, clear_after_read=False,
                      element_shape=())
        self._crit_i = 0
    

class RewardPenaltyLogger:

    def __init__(self, save_dir, expected_num_logs=1000, fig_name="Rewards and Penalties", runavg_interval=10):
        self.exp_logs = expected_num_logs
        self.reset()
        self.fig_name = fig_name
        self.save_dir = save_dir
        self.runavg_interval = runavg_interval
    
    def add_reward(self, r):
        self._r_ta = self._r_ta.write(self._r_i, r)
        self._r_i += 1
    
    def add_penalty(self, p):
        self._p_ta = self._p_ta.write(self._p_i, p)
        self._p_i += 1
    
    def export(self):
        return {"rewards" : reshape(self._r_ta.gather(range(0,self._r_i)), [-1]).numpy(),
                "penalties" : reshape(self._p_ta.gather(range(0,self._p_i)), [-1]).numpy()}
    
    def export_as_running_avg(self, interval=10):
        export = self.export()
        r_avg = np.empty_like(export["rewards"])
        for i in range(len(r_avg)):
            r_avg[i] = np.sum(export["rewards"][max(0, i-interval):i+1]) / (i + 1 - max(0, i-interval))
        p_avg = np.empty_like(export["penalties"])
        for i in range(len(p_avg)):
            p_avg[i] = np.sum(export["penalties"][max(0, i-interval):i+1]) / (i + 1 - max(0, i-interval+1))
        return {"rewards_runavg" : r_avg, "penalties_runavg" : p_avg}
    
    def save_rew_pen_plots(self, name_suffix=""):
        
        fig = plt.figure(layout="constrained", figsize=(6,4), dpi=100)
        fig.suptitle(self.fig_name, fontsize=12)
        axR = plt.subplot2grid((1,2), (0,0))
        axP = plt.subplot2grid((1,2), (0,1))

        axR.set_title("Rewards")
        axP.set_title("Penalties")

        export = self.export()
        avg_export = self.export_as_running_avg(self.runavg_interval)

        axR.plot(export["rewards"], '-r')
        axR.plot(avg_export["rewards_runavg"], '--g')

        axP.plot(export["penalties"], '-r')
        axP.plot(avg_export["penalties_runavg"], '--g')

        if name_suffix != "": name_suffix = "_" + name_suffix

        fig.savefig(ospath_join(self.save_dir, "rews_pens" + name_suffix))
        plt.close(fig)
    
    def reset(self):
        if hasattr(self, "_r_ta"): self._r_ta.close()
        if hasattr(self, "_p_ta"): self._p_ta.close()
        self._r_ta = ta(dtype=tfFloat32, size=self.exp_logs,
                      dynamic_size=True, clear_after_read=False,
                      element_shape=())
        self._r_i = 0
        self._p_ta = ta(dtype=tfFloat32, size=self.exp_logs,
                      dynamic_size=True, clear_after_read=False,
                      element_shape=())
        self._p_i = 0