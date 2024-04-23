import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
from PhysicalEnvironment.StateTrajectory import TrajectoryExport
from os.path import join as ospath_join



class Visualizer:

    def __init__(self, save_dir, n_vehicle_shows=5):
        self.vehicle_show_count = n_vehicle_shows
        self.save_dir = save_dir

    @staticmethod
    def vehicle_plot_triangle(x_center, y_center, phi, height, base):
        return [[x_center+np.cos(phi)*height/2,
                x_center-np.cos(phi)*height/2-np.cos(np.pi/2-phi)*base/2,
                x_center-np.cos(phi)*height/2+np.cos(np.pi/2-phi)*base/2,
                x_center+np.cos(phi)*height/2],
                [y_center+np.sin(phi)*height/2,
                 y_center-np.sin(phi)*height/2+np.sin(np.pi/2-phi)*base/2,
                 y_center-np.sin(phi)*height/2-np.sin(np.pi/2-phi)*base/2,
                 y_center+np.sin(phi)*height/2]]

    def visualize_state_trajectory(self, tr_ex:TrajectoryExport,
                                   controller_info:dict|None=None, simulation_info:dict|None=None,
                                   training_info:dict|None=None, reward_info:dict|None=None,
                                   n_vehicle_markers=None, file_name=None, fig_name:str=None):
        
        if n_vehicle_markers is None: n_vehicle_markers = self.vehicle_show_count
        if file_name is None and fig_name is None: file_name = "fig"
        if fig_name is None: fig_name = file_name

        tr_ex.return_as_np = True

        fig = plt.figure(layout="constrained", figsize=(15,10), dpi=100)
        fig.suptitle(fig_name, fontsize=24)
        axS = plt.subplot2grid((8,12), (0,0), rowspan=6, colspan=6)
        axU = plt.subplot2grid((8,12), (0,6), rowspan=2, colspan=3)
        axW = plt.subplot2grid((8,12), (2,6), rowspan=2, colspan=3)
        axR = plt.subplot2grid((8,12), (4,6), rowspan=2, colspan=3)
        axE = plt.subplot2grid((8,12), (0,9), rowspan=2, colspan=3)
        axA = plt.subplot2grid((8,12), (2,9), rowspan=2, colspan=3)
        axT = plt.subplot2grid((8,12), (4,9), rowspan=2, colspan=3)
        txt = plt.subplot2grid((8,12), (6,0), rowspan=2, colspan=12)

        axS.set_title("State Space")
        axW.set_title("Angular Drive")
        axU.set_title("Linear Drive")
        axR.set_title("Rewards Yielded")
        axE.set_title("State \'e\'")
        axA.set_title("State \'\u03B1\'")
        axT.set_title("State \'\u03B8\'")

        s_bounds = {"x_min": np.min(tr_ex.cartesian_states[:,0]), "x_max": np.max(tr_ex.cartesian_states[:,0]),
                    "y_min": np.min(tr_ex.cartesian_states[:,1]), "y_max": np.max(tr_ex.cartesian_states[:,1])}
        x_width = s_bounds["x_max"] - s_bounds["x_min"]
        y_width = s_bounds["y_max"] - s_bounds["y_min"]
        if x_width > y_width:
            s_bounds["y_min"] = s_bounds["y_min"] - (x_width-y_width)/2
            s_bounds["y_max"] = s_bounds["y_max"] + (x_width-y_width)/2
            y_width = x_width
        else:
            s_bounds["x_min"] = s_bounds["x_min"] - (y_width-x_width)/2
            s_bounds["x_max"] = s_bounds["x_max"] + (y_width-x_width)/2
            x_width = y_width
        s_bounds["x_min"] = s_bounds["x_min"] - x_width / 10
        s_bounds["x_max"] = s_bounds["x_max"] + x_width / 10
        s_bounds["y_min"] = s_bounds["y_min"] - y_width / 10
        s_bounds["y_max"] = s_bounds["y_max"] + y_width / 10
        s_width = (x_width + y_width) / 2
        
        #vehicle points:
        step = tr_ex.point_count // (n_vehicle_markers - 1)
        v_idxs = list(range(0, tr_ex.point_count, step))
        if tr_ex.point_count % (n_vehicle_markers - 1) != 0:
            v_idxs[-1] = -1
        else:
            v_idxs.append(-1)
        #tri_height = tr_ex.length / n_vehicle_markers / 8
        tri_height = s_width * 1.337 * 3.0 / n_vehicle_markers / 8

        axS.plot(tr_ex.cartesian_states[:,0], tr_ex.cartesian_states[:,1], '-b')
        vehicle_markers = [self.vehicle_plot_triangle(tr_ex.cartesian_states[idx,0],
                                                      tr_ex.cartesian_states[idx,1],
                                                      tr_ex.cartesian_states[idx,2],
                                                      tri_height,
                                                      tri_height/1.5) for idx in v_idxs]
        for tri in vehicle_markers:
            axS.plot(tri[0], tri[1], color=(1.0,0.0,0.0,0.8), zorder=1, linewidth=1.0)
        axS.plot(0.0, 0.0, color=(0.0,0.0,0.0,0.6), marker='x', markersize=50.0, zorder=0, linewidth=2.2)

        #axS.set_xlabel("x", loc="left")
        #axS.set_ylabel("y", loc="bottom")
        axS.set_xbound(s_bounds["x_min"], s_bounds["x_max"])
        axS.set_ybound(s_bounds["y_min"], s_bounds["y_max"])
        axS.grid(True, which='both', axis='both', color=(0.1,0.1,0.6,0.4), linestyle='-', linewidth=0.3, zorder=-1)
        axS.axhline(y=0.0, xmin=s_bounds["x_min"], xmax=s_bounds["x_max"], color=(0.0,0.0,0.0,0.5), linestyle='-', linewidth=0.35, zorder=0)
        axS.axvline(x=0.0, ymin=s_bounds["x_min"], ymax=s_bounds["x_max"], color=(0.0,0.0,0.0,0.5), linestyle='-', linewidth=0.35, zorder=0)

        u_bounds = {"x_min": np.min(tr_ex.time), "x_max": np.max(tr_ex.time),
                    "y_min": np.min(tr_ex.control_inputs[:,0]), "y_max": np.max(tr_ex.control_inputs[:,0])}
        u_bounds["x_min"] = u_bounds["x_min"] - (u_bounds["x_max"] - u_bounds["x_min"]) / 16
        u_bounds["x_max"] = u_bounds["x_max"] + (u_bounds["x_max"] - u_bounds["x_min"]) / 17
        u_bounds["y_min"] = u_bounds["y_min"] - (u_bounds["y_max"] - u_bounds["y_min"]) / 10
        u_bounds["y_max"] = u_bounds["y_max"] + (u_bounds["y_max"] - u_bounds["y_min"]) / 11

        w_bounds = {"x_min" : u_bounds["x_min"], "x_max" : u_bounds["x_max"],
                    "y_min": np.min(tr_ex.control_inputs[:,1]), "y_max": np.max(tr_ex.control_inputs[:,1])}
        w_bounds["y_min"] = w_bounds["y_min"] - (w_bounds["y_max"] - w_bounds["y_min"]) / 10
        w_bounds["y_max"] = w_bounds["y_max"] + (w_bounds["y_max"] - w_bounds["y_min"]) / 11

        veh_vlines_kwargs = {"colors":(1.0,0.0,0.0,0.5), "linewidths":0.7, "zorder":1}

        axU.plot(tr_ex.time[:-1], tr_ex.control_inputs[:,0][:-1], color='tab:orange')
        axU.vlines(x=tr_ex.time[v_idxs], ymin=u_bounds["y_min"], ymax=u_bounds["y_max"], **veh_vlines_kwargs)
        axU.set_xbound(u_bounds["x_min"], u_bounds["x_max"])
        axU.set_ybound(u_bounds["y_min"], u_bounds["y_max"])
        axU.grid(True, which='both', axis='y', color=(0.1,0.1,0.6,0.4), linestyle='-', linewidth=0.25, zorder=0)
        
        axW.plot(tr_ex.time[:-1], tr_ex.control_inputs[:,1][:-1]/np.pi, color='tab:orange')
        axW.yaxis.set_major_formatter(tck.FormatStrFormatter('%g$\pi$'))
        axW.yaxis.set_major_locator(tck.AutoLocator())
        axW.vlines(x=tr_ex.time[v_idxs], ymin=w_bounds["y_min"], ymax=w_bounds["y_max"], **veh_vlines_kwargs)
        axW.set_xbound(w_bounds["x_min"], w_bounds["x_max"])
        axW.set_ybound(w_bounds["y_min"]/np.pi, w_bounds["y_max"]/np.pi)
        axW.grid(True, which='both', axis='y', color=(0.1,0.1,0.6,0.4), linestyle='-', linewidth=0.25, zorder=0)

        axR.plot(tr_ex.time[:-1], tr_ex.rewards[:-1])

        e_bounds = {"x_min" : u_bounds["x_min"], "x_max" : u_bounds["x_max"],
                    "y_min" : 0.0, "y_max" : 1.1*np.max(tr_ex.polar_states[:,0])}
        axE.plot(tr_ex.time, tr_ex.polar_states[:,0], "-b")
        axE.vlines(x=tr_ex.time[v_idxs], ymin=e_bounds["y_min"], ymax=e_bounds["y_max"], **veh_vlines_kwargs)
        axE.set_xbound(e_bounds["x_min"], e_bounds["x_max"])
        axE.set_ybound(e_bounds["y_min"], e_bounds["y_max"])

        axA.plot(tr_ex.time, tr_ex.polar_states[:,1]/np.pi, "-b")
        axA.yaxis.set_major_formatter(tck.FormatStrFormatter('%g$\pi$'))
        axA.yaxis.set_major_locator(tck.AutoLocator())
        axA.vlines(x=tr_ex.time[v_idxs], ymin=-1, ymax=1, **veh_vlines_kwargs)
        axA.axhline(y=0.0, xmin=u_bounds["x_min"], xmax=u_bounds["x_max"], color=(0.1,0.1,0.6,0.4), linestyle='-', linewidth=0.3, zorder=0)
        axA.set_xbound(u_bounds["x_min"], u_bounds["x_max"])
        axA.set_ybound(-1, 1)

        axT.plot(tr_ex.time, tr_ex.polar_states[:,2]/np.pi, "-b")
        axT.yaxis.set_major_formatter(tck.FormatStrFormatter('%g$\pi$'))
        axT.yaxis.set_major_locator(tck.AutoLocator())
        axT.vlines(x=tr_ex.time[v_idxs], ymin=-1, ymax=1, **veh_vlines_kwargs)
        axA.axhline(y=0.0, xmin=u_bounds["x_min"], xmax=u_bounds["x_max"], color=(0.1,0.1,0.6,0.4), linestyle='-', linewidth=0.3, zorder=0)
        axT.set_xbound(u_bounds["x_min"], u_bounds["x_max"])
        axT.set_ybound(-1, 1)

        txt.axis([0,24,0,4])
        txt.set_axis_off()
        
        if simulation_info:
            sim_text = "Simulation Info| "
            for key in simulation_info:
                sim_text = sim_text + key + ": " + str(simulation_info[key]) + ", "
            sim_text = sim_text[:-2]
            sim_text = sim_text + "."
            txt.text(0.1, 3.4, sim_text, fontsize=10)
        
        #if tr_ex.length.shape == ():
        #    l_tr = str(tr_ex.length)
        #else:
        #    l_tr = str(tr_ex.length.flatten()[0])
        l_tr = str(tr_ex.length)[:4]
        t_tr = str(tr_ex.time[-1].flatten())[1:-1]
        for i, char in enumerate(t_tr, start=1):
            if char == '.':
                t_tr = t_tr[:min(i+1, len(t_tr))]
                break
        if tr_ex.reached_goal:
            g_tr = "yes"
        else:
            g_tr = "no"
        trajectory_infotext = "Trajectory Info| Reached Goal: {yesgoal}, Length: {length}, Total Time: {time}.".format(length=l_tr, yesgoal=g_tr, time=t_tr)
        txt.text(0.1, 2.8, trajectory_infotext, fontsize=10)

        if controller_info:
            controller_text = "Controller Info| "
            for key in controller_info:
                controller_text = controller_text + key + ": " + str(controller_info[key]) + ", "
            controller_text = controller_text[:-2]
            controller_text = controller_text + "."
            txt.text(0.1, 2.2, controller_text, fontsize=10)
        
        if training_info:
            training_text = "Training Info| "
            for key in training_info:
                training_text = training_text + key + ": " + str(training_info[key]) + ", "
            training_text = training_text[:-2]
            training_text = training_text + "."
            txt.text(0.1, 1.6, training_text, fontsize=10)
        
        if reward_info:
            reward_text = "Reward Info| "
            for key in reward_info:
                reward_text = reward_text + key + ": " + str(reward_info[key]) + ", "
            reward_text = reward_text[:-2]
            reward_text = reward_text + "."
            txt.text(0.1, 1.0, reward_text, fontsize=10)

        fig.savefig(ospath_join(self.save_dir, file_name))
        plt.close(fig)