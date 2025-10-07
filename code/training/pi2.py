import os
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.spatial import distance

from dmp import dmp_cartesian as dmp

class PI2:
    def __init__(self):
        self.dmp = None
        self.plot_dmp_init = True
        self.covar = None
        self.cov_decay = 1
        self.sigma = 0.001#0.5
        self.sigma_min = 0.0001
        self.sigma_max = 0.05
        self.sigma_steepness = 2
        self.sigma_goal = 0.003
        self.covar_goal = None
        self.bounds = [0.003,0.003] #[0.005,0.005]
        self.learned_L = None
        self.param_pertub = None
        self.n_bfs_p = 0
        self.tau = 1.0
        self.plot_frequency = 50


    def initialize_dmp(self,trajectory_file,n_dmps=3, n_bfs=10,K=25,rescale='rotodilatation', basis='rbf', alpha_s=4.0, tol=0.001):
        self.n_bfs_p = n_bfs + 1 #dmp library adds one to the specified number of basis functions
        trajectory = np.loadtxt(trajectory_file, delimiter=',')  # Adjust the delimiter if necessary
        print(trajectory.shape)
        T = trajectory[-1, 0] - trajectory[0, 0]
        n_traj_steps, _ = trajectory.shape
        dt = T/n_traj_steps
        t_traj = trajectory[:, 0]
        xs_traj = trajectory[:, 1:1 + n_dmps]
        xds_traj = trajectory[:, 1 + n_dmps:1 + 2 * n_dmps]
        xdds_traj = trajectory[:, 1 + 2 * n_dmps:1 + 3 * n_dmps]
        x_0 = xs_traj[0, :] # + self.init_pertub
        x_goal = xs_traj[-1, :] # + self.goal_pertub

        self.dmp = dmp.DMPs_cartesian(
            n_dmps = n_dmps,
            n_bfs = n_bfs,
            #dt = dt,
            x_0 = x_0,
            x_goal = x_goal,
            #T = T,
            K = K,
            rescale = rescale,
            alpha_s = alpha_s,
            tol = tol,
            basis=basis,
            h_=0.5)

        self.get_covar()
        f_target = self.dmp.imitate_path(x_des=xs_traj,dx_des=xds_traj,ddx_des=xdds_traj, t_des =t_traj.copy())
        self.learned_L = np.linalg.norm(self.dmp.learned_position)

        x_learned, dx_learned, ddx_learned, t_learned, f_track, weighted_ac = self.dmp.rollout(tau=self.tau)

        if self.plot_dmp_init:
            for i in range(self.dmp.n_dmps):
                if i == 1:

                    plt.figure(num=i,figsize=(13, 8), dpi=100)

                    plt.plot(f_target[i, :], color=[0, 0.7, 0], linewidth=2)
                    plt.plot(np.sum(weighted_ac, axis=1)[:,i], 'k', linewidth=1.5)
                    plt.plot(weighted_ac[:,i,:], 'k:', linewidth=1.5)
                    plt.xlabel("Time Steps")
                    plt.title("Forcing Terms")
                    plt.legend(["Target", "Approximation", "Basis Functions"])
                    tau = 15
                    ts = np.linspace(0, tau, len(trajectory))
                    #print(t_traj)
                    #ts_exp = self.xs_phase * self.tau
                    plt.figure(figsize=(13, 8), dpi=100)
                    plt.suptitle("The demonstrated Trajectory and its Reproduction")
                    plt.subplot(3, 1, 1)
                    plt.plot(t_traj, xs_traj, 'b', linewidth=1)
                    #plt.plot(ts[:-1], m[:, 1:1 + self.dmp.n_dmps], 'k')
                    plt.plot(t_learned, x_learned, 'k')
                    plt.xlabel("Time in s")
                    plt.ylabel("Position")
                    plt.legend(["Demonstration in x, y, z", "","","Reproduction in x, y, z"])
                    plt.subplot(3, 1, 2)
                    plt.plot(t_traj, xds_traj, 'b')
                    plt.plot(t_learned, dx_learned, 'k')
                    plt.xlabel("Time in s")
                    plt.ylabel("Velocity")
                    plt.subplot(3, 1, 3)
                    plt.plot(t_traj, xdds_traj, 'b')
                    plt.plot(t_learned, ddx_learned, 'k')
                    plt.xlabel("Time in s")
                    plt.ylabel("Acceleration")
                    plt.show()

        return f_target

    def optimize_2D_3D_plot(self, max_runs, n_samples, z_vec_high, z_vec_low, x_vec_high, covar_decay_factor, saving, directory, max_val_z, min_val_z, max_val_x,weights_z,mode="", axes=None,optimize_counter = 0,visualize_progress=True):
        h = 10  # label_zavg_costs
        train_data = np.zeros((max_runs, self.n_bfs_p, self.dmp.n_dmps))
        label_z = np.zeros((max_runs, len(z_vec_high)))
        y_locations_z = np.zeros((max_runs, len(z_vec_high)))
        y_locations_x = np.zeros((max_runs, len(x_vec_high)))
        label_x = np.zeros((max_runs, len(x_vec_high)))
        start_id = -1 #when the constraint are satified, beginn collecting the trajectory shape data
        started = False
        avg_costs = np.zeros((max_runs, 6)) #six cost parts here!
        gm1 = 0
        gm2 = 0
        gm3 = 0
        ti_start = time.time()

        if axes is not None:
            ax1 = axes[0]
            #ax2 = axes[1]

            if len(axes) > 4:
                ax2 = axes[1]
                ax3 = axes[2]
                ax6 = axes[3]
                ax7 = axes[4]
                ax8 = axes[5]
            else:
                ax6 = axes[1]
                ax7 = axes[2]
                ax8 = axes[3]

        else:
            fig1, ax1 = plt.subplots(figsize=(8, 8))
            fig_manager = plt.get_current_fig_manager()
            # Set the position of the figure on the screen (adjust values accordingly)
            #fig_manager.window.wm_geometry("+0+100")
            fig2, (ax6, ax7) = plt.subplots(1, 2,figsize=(10, 4), facecolor='w')

        ax1.grid(False)

        ax1.set_xlim([-0.03, 0.1])
        ax1.set_ylim([-0.02, 0.17])
        ax1.set_zlim([-0.01, 0.1]) #0.22
        ax1.set_xlabel(r"$\mathbf{e}_2$ [m]")
        ax1.set_ylabel(r"$\mathbf{e}_1$ [m]")
        ax1.set_zlabel(r"$\mathbf{e}_3$ [m]")

        ax2.grid(False)
        zlh = ax2.set_ylabel(r"$\mathbf{e}_3$ [m]", labelpad=10)

        for iii in range(len(z_vec_high)):
            ax2.plot(z_vec_high[iii],max_val_z,'b*')

        L = ax2.plot([-0.05, 0.25], [0, 0], color='k', linestyle='--', linewidth=1)[0]
        ax2.plot([-0.05, 0.25], [0, 0], color='k', linewidth=0.5)
        H = ax2.text(0, -0.042, '0', fontsize=12)
        ax2.axis('auto')
        ax2.set_xlim([-0.02, 0.17])
        ax2.set_ylim([-0.01, max_val_z+0.07]) #0.22
        ax2.set_xlabel(r"$\mathbf{e}_1$ [m]")
        ax2.plot([z_vec_low[0], z_vec_low[1]], [min_val_z+0.002, min_val_z+0.002], color='r', linestyle='--', linewidth=1)[0]
        ax2.plot([z_vec_low[0], z_vec_low[1]], [min_val_z-0.002, min_val_z-0.002], color='r', linestyle='--', linewidth=1)[0]

        ax3.grid(False)
        zlh = ax3.set_ylabel(r"$\mathbf{e}_2$ [m]", labelpad=10)

        for iii in range(len(x_vec_high)):
            ax3.plot(x_vec_high[iii],max_val_x,'bo')

        ax3.axis('auto')
        ax3.set_xlim([-0.02, 0.17])
        ax3.set_ylim([-0.01, max_val_z+0.07]) #0.22
        ax3.set_xlabel(r"$\mathbf{e}_1$ [m]")

        ax6.set_xlabel(r"$\theta_{i,\mathbf{e}_1}$")
        ax6.set_ylabel("Activations")
        plt.suptitle("Forcing term parameters at iteration j=0")
        ax7.set_xlabel(r"$\theta_{i,\mathbf{e}_2}$")
        ax8.set_xlabel(r"$\theta_{i,\mathbf{e}_3}$")


        for ii in range(max_runs):
            mean_expl = self.explore(n_samples)
            costs = np.zeros((n_samples, 6)) #six costs here!!

            if saving:
                if not os.path.exists(directory):
                    os.makedirs(directory)
                os.makedirs(directory + "/run_" + str(ii))

            for jj in range(n_samples):
                means_e = mean_expl[jj, :, :]
                #print("means e", means_e, "reshaped: ",np.reshape(means_e, [self.dmp.n_dmps,self.n_bfs_p]), "transposed: ", means_e.transpose())
                self.dmp.w = means_e.transpose()
                x_track, dx_track, ddx_track, t_track, _, _ = self.dmp.rollout(tau=self.tau)
                costs[jj, :], _, _, _, _, _, _, _, _ = self.evaluate_rollout_2D(x_track, ddx_track, z_vec_high, z_vec_low,x_vec_high, max_val_z,min_val_z,max_val_x,weights_z,mode)

            weights = self.costs_to_weights(costs[:, 0], h)
            mean_new = np.mean(mean_expl * weights[:, np.newaxis, np.newaxis], axis=0) / np.mean(weights)
            mean_new = np.reshape(mean_new, [self.n_bfs_p,self.dmp.n_dmps])
            mean_old = self.dmp.w
            self.dmp.w = mean_new.transpose()
            self.covar = covar_decay_factor**2 * self.covar
            x_track, dx_track, ddx_track, t_track, _, _ = self.dmp.rollout(tau=self.tau)

            policy_cost, label_z[ii, :], label_x[ii, :], start,fin, _, _, y_borders_z, y_borders_x = self.evaluate_rollout_2D(x_track, ddx_track, z_vec_high, z_vec_low,x_vec_high, max_val_z,min_val_z,max_val_x,weights_z,mode)
            if not started:
                if start:
                    start_id = ii
                    started = True #get first id when constraints are satified
                    print('start at id: ', start_id)
            train_data[ii, :, :] = mean_new
            avg_costs[ii, :] = policy_cost
            y_locations_z[ii, :] = y_borders_z
            y_locations_x[ii, :] = y_borders_x

            if ii % self.plot_frequency == 0 or fin or ii == max_runs:
                #ticks = [0] + y_borders_z.tolist() + y_borders_x.tolist() + [0.15]
                ax1.set_yticks([0] + y_borders_z.tolist() + y_borders_x.tolist() + [0.15])
                ax1.set_yticklabels(['0', r'$p_1$',r'$p_2$', r'$p_3$', r'$p_4$',r'$p_5$',r'$p_6$','0.15'], fontsize=12)

                ax2.set_xticks([0] + y_borders_z.tolist() + [0.15])
                ax2.set_xticklabels(['0', r'$p_1$',r'$p_2$', r'$p_3$', r'$p_4$','0.15'], fontsize=12)

                s_1 = -policy_cost[1]/np.min(weights_z)
                new_values_L = [-0.05, 0.25], [s_1, s_1]
                new_value_H = [-0.02, s_1, '0']

                L.set_xdata(new_values_L[0])
                L.set_ydata(new_values_L[1])

                H.set_text(r"$s_1=$"+str(np.round(s_1,4)))
                H.set_position((new_value_H[0], new_value_H[1]))

                f_c = s_1/max_val_z
                f_c = max(0, min(f_c, 1))

                self.plot_rollout(ax2,[0,0,f_c],1)

                ax3.set_xticks([0, y_borders_x[0], y_borders_x[1], 0.15])
                ax3.set_xticklabels(['0', r'$p_5$',r'$p_6$', '0.15'], fontsize=12)
                self.plot_rollout_x(ax3,[0,0,f_c],1)
                plt.suptitle("DMP trajectory at iteration j=" + str(ii))


                f_c = -policy_cost[1]/max_val_z/np.min(weights_z)
                f_c = max(0, min(f_c, 1))

                self.plot_rollout_3d(ax1,[0,0,f_c],1)
                ax1.set_title("3D DMP trajectory at iteration j=" + str(ii))

                ax6.clear()
                    #ax6 = fig2.add_subplot(1, 2, 1)
                ax6.set_xlabel(r"$\theta_{i,\mathbf{e}_1}$")
                ax6.set_ylabel("Activations")
                ax6.set_ylim([-10, 12])
                plt.suptitle("Forcing term parameters at iteration j=" + str(ii))

                ax7.clear()
                #ax7 = fig2.add_subplot(1, 2, 2)
                ax7.set_xlabel(r"$\theta_{i,\mathbf{e}_2}$")
                #ax7.set_ylabel("Activations")
                ax7.set_ylim([-10, 12])

                ax8.clear()
                #ax7 = fig2.add_subplot(1, 2, 2)
                ax8.set_xlabel(r"$\theta_{i,\mathbf{e}_3}$")
                #ax7.set_ylabel("Activations")
                ax8.set_ylim([-10, 12])

                # Assuming a loop to calculate new values for bar plots
                # Replace the following lines with your logic to get new bar plot values
                new_means_2 = self.dmp.w[1, :]
                new_means_1 = self.dmp.w[0, :]
                new_means_3 = self.dmp.w[2, :]




                ba1 = ax6.bar(range(len(new_means_2)), new_means_2, color=[0,0,f_c])
                ba2 = ax7.bar(range(len(new_means_1)), new_means_1, color=[0,0,f_c])
                ba3 = ax8.bar(range(len(new_means_3)), new_means_3, color=[0,0,f_c])
                #fig2.canvas.draw()
                #print("optimization counter: ", ii, "policy cost: ", policy_cost)
                if visualize_progress:
                    plt.pause(0.01)


            if ii == max_runs-1 or fin:
                ti_end = time.time()
                ti = ti_end-ti_start
                print(f'finished optimization at iteration {ii} after {ti:2f}s, mean_z_0 = {mean_new[0,2]}')
                break
        plt.ion() #allows the execution of code after the plt.show(), e.g. the closing of plots
        plt.show()
        #fig_handle = plt.gcf()
        return avg_costs, h, train_data, label_z, label_x, start_id,ii, ti, y_locations_z, y_locations_x

    def optimize_continuous_box_acc(self, max_runs, n_samples, z_vec, covar_decay_factor, saving, directory, max_val, mode="", axes=None,optimize_counter = 0,visualize_progress=0):
        h = 10  # label_zavg_costs
        train_data = np.zeros((max_runs, self.n_bfs_p, self.dmp.n_dmps))
        label_z = np.zeros((max_runs, len(z_vec)))
        y_locations = np.zeros((max_runs, len(z_vec)))
        label_x = label_z

        avg_costs = np.zeros((max_runs, 5))
        gm1 = 0
        gm2 = 0
        gm3 = 0
        ti_start = time.time()

        if axes is not None:
            ax1 = axes[0]
            ax6 = axes[1]
            ax7 = axes[2]

        else:
            fig1, ax1 = plt.subplots(figsize=(8, 8))
            fig_manager = plt.get_current_fig_manager()
            # Set the position of the figure on the screen (adjust values accordingly)
            fig_manager.window.wm_geometry("+0+100")
            fig2, (ax6, ax7) = plt.subplots(1, 2,figsize=(10, 4), facecolor='w')


        ax1.grid(False)
        zlh = ax1.set_ylabel('Height-to-length ratio r_c', labelpad=10)

        ax1.plot([-self.bounds[0], -self.bounds[0]], [0, 0.25], color='r', linestyle='--', linewidth=1)
        ax1.plot([self.learned_L+self.bounds[1], self.learned_L+self.bounds[1]], [0, 0.25], color='r', linestyle='--', linewidth=1)
        for iii in range(len(z_vec)):
            ax1.plot(z_vec[iii],max_val,'b*')
        L = ax1.plot([-0.05, 0.25], [0, 0], color='k', linestyle='--', linewidth=1)[0]
        ax1.plot([-0.05, 0.25], [0, 0], color='k', linewidth=0.5)
        H = ax1.text(0, -0.042, '0', fontsize=12)
        ax1.axis('auto')
        ax1.set_xlim([-0.02, 0.18])
        ax1.set_ylim([-0.01, max_val+0.02]) #0.22
        ax1.set_xlabel(r"$\mathbf{e}_1}$ [m]")
        ax1.set_ylabel(r"$\mathbf{e}_2}$ [m]")
        ax1.set_title("DMP trajectory at iteration j=0")
        ax6.set_xlabel(r"$\theta_{i,\mathbf{e}_1}}$")
        ax6.set_ylabel("Activations")
        ax7.set_xlabel(r"$\theta_{i,\mathbf{e}_2}$")
        plt.suptitle("Forcing term parameters at iteration j=0")

        for ii in range(max_runs):
            #print("optimization counter: ", ii)
            mean_expl = self.explore(n_samples)
            costs = np.zeros((n_samples, 5))

            if saving:
                if not os.path.exists(directory):
                    os.makedirs(directory)
                os.makedirs(directory + "/run_" + str(ii))

            #start_time = time.time()
            for jj in range(n_samples):
                means_e = mean_expl[jj, :, :]
                #means_e = np.reshape(means_e, [self.n_bfs_p, self.dmp.n_dmps])
                #print("means e", means_e, "reshaped: ",np.reshape(means_e, [self.dmp.n_dmps,self.n_bfs_p]), "transposed: ", means_e.transpose())
                self.dmp.w = means_e.transpose()
                x_track, dx_track, ddx_track, t_track, _, _ = self.dmp.rollout(tau=self.tau)
                #m = [t_track, x_track, dx_track, ddx_track]
                costs[jj, :], _, _, _, _ = self.evaluate_rollout_continuous_box_acc(x_track, ddx_track, z_vec, max_val, mode)

            weights = self.costs_to_weights(costs[:, 0], h)
            mean_new = np.mean(mean_expl * weights[:, np.newaxis, np.newaxis], axis=0) / np.mean(weights)
            mean_new = np.reshape(mean_new, [self.n_bfs_p,self.dmp.n_dmps])
            mean_old = self.dmp.w
            self.dmp.w = mean_new.transpose()
            self.covar = covar_decay_factor**2 * self.covar
            #start_time = time.time()
            x_track, dx_track, ddx_track, t_track, _, _ = self.dmp.rollout(tau=self.tau)

            policy_cost, label_z[ii, :], fin, y_borders, _ = self.evaluate_rollout_continuous_box_acc(x_track, ddx_track, z_vec, max_val, mode)

            train_data[ii, :, :] = mean_new
            avg_costs[ii, :] = policy_cost
            y_locations[ii, :] = y_borders

            if ii % self.plot_frequency == 0 or fin or ii == max_runs:

                ax1.set_xticks([0] + y_borders.tolist() + [0.15])
                ax1.set_xticklabels(['0'] + [f'p{i+1}' for i in range(len(y_borders))] + ['0.15'], fontsize=12)

                [ax1.plot([y, y], [0,-policy_cost[1]], color='g', linestyle=':', linewidth=2)[0] for y in y_borders]


                new_values_L = [-0.05, 0.25], [-policy_cost[1], -policy_cost[1]]
                #r_c = np.round(-policy_cost[1]/0.15,2)
                new_value_H = [-0.02, -policy_cost[1], '0']


                L.set_xdata(new_values_L[0])
                L.set_ydata(new_values_L[1])
                    #L.set_3d_properties(new_values_L[2])

                H.set_text(r"$s_1=$"+str(np.round(-policy_cost[1],4)))
                H.set_position((new_value_H[0], new_value_H[1]))

                f_c = -policy_cost[1]/max_val
                f_c = max(0, min(f_c, 1))

                self.plot_rollout(ax1,[0,0,f_c],1)
                ax1.set_title("DMP trajectory at iteration j=" + str(ii))
                ax6.clear()
                    #ax6 = fig2.add_subplot(1, 2, 1)
                ax6.set_xlabel(r"$\theta_{i,\mathbf{e}_1}}$")
                ax6.set_ylabel("Activations")
                ax6.set_ylim([-5.5, 12])

                plt.suptitle("Forcing term parameters at iteration j=" + str(ii),y=0.92)

                ax7.clear()
                #ax7 = fig2.add_subplot(1, 2, 2)
                ax7.set_xlabel(r"$\theta_{i,\mathbf{e}_2}}$")
                #ax7.set_ylabel("Activations")
                ax7.set_ylim([-5.5, 12])

                new_means_2 = self.dmp.w[1, :]
                new_means_3 = self.dmp.w[2, :]




                ba1 = ax6.bar(range(len(new_means_2)), new_means_2, color=[0,0,f_c])
                ba2 = ax7.bar(range(len(new_means_3)), new_means_3, color=[0,0,f_c])

                if visualize_progress:
                    plt.pause(0.1)


            if ii == max_runs or fin:
                print("final costs, accelerations, jerk, and first acceleration: ",policy_cost,np.linalg.norm(ddx_track),np.linalg.norm(np.diff(ddx_track,axis=0)),np.sum(np.absolute(ddx_track[0,:])))
                ti_end = time.time()
                ti = ti_end-ti_start
                print(f'finished optimization at iteration {ii} after {ti:2f}s, n_traj_points = {len(x_track)}')
                break
        plt.ion() #allows the execution of code after the plt.show(), e.g. the closing of plots
        plt.show()
        #fig_handle = plt.gcf()
        return avg_costs, h, train_data, label_z, ii, ti, y_locations, x_track

    def optimize_one_param_acc(self, max_runs, n_samples, covar_decay_factor, saving, directory, max_val, mode="", axes=None,optimize_counter = 0,visualize_progress=0):
        h = 10  # label_zavg_costs
        train_data = np.zeros((max_runs, self.n_bfs_p, self.dmp.n_dmps))
        label_z = np.zeros(max_runs)
        label_x = label_z

        avg_costs = np.zeros((max_runs, 5))
        gm1 = 0
        gm2 = 0
        gm3 = 0
        ti_start = time.time()

        if axes is not None:
            ax1 = axes[0]
            ax6 = axes[1]
            ax7 = axes[2]

        else:
            fig1, ax1 = plt.subplots(figsize=(8, 8))
            fig_manager = plt.get_current_fig_manager()
            # Set the position of the figure on the screen (adjust values accordingly)
            fig_manager.window.wm_geometry("+0+100")
            fig2, (ax6, ax7) = plt.subplots(1, 2,figsize=(10, 4), facecolor='w')

        ax1.grid(False)

        H = ax1.text(0, -0.042, '0', fontsize=12)
        ax1.axis('auto')
        ax1.set_xlim([-0.02, 0.17])
        ax1.set_ylim([-0.01, max_val+0.07]) #0.22
        ax1.set_xlabel(r"$\mathbf{e}_1}$ [m]")
        ax1.set_ylabel(r"$\mathbf{e}_2}$ [m]")

        circle_patch = plt.Circle([0.075, 0.0], 0.00, edgecolor='white', linestyle='dashed', linewidth=2, fill=False, zorder=10)
        ax1.add_patch(circle_patch)
        ax6.set_xlabel(r"$\theta_{i,\mathbf{e}_1}}$")
        ax6.set_ylabel("Activations")
        plt.suptitle("Forcing term parameters at iteration j=0")
        ax7.set_xlabel(r"$\theta_{i,\mathbf{e}_2}$")


        for ii in range(max_runs):
            mean_expl = self.explore(n_samples)
            costs = np.zeros((n_samples, 5))

            if saving:
                if not os.path.exists(directory):
                    os.makedirs(directory)
                os.makedirs(directory + "/run_" + str(ii))

            for jj in range(n_samples):
                means_e = mean_expl[jj, :, :]
                self.dmp.w = means_e.transpose()
                x_track, dx_track, ddx_track, t_track, _, _ = self.dmp.rollout(tau=self.tau)
                costs[jj, :], _, _ = self.evaluate_rollout_one_param_acc(x_track, ddx_track,max_val)


            weights = self.costs_to_weights(costs[:, 0], h)
            mean_new = np.mean(mean_expl * weights[:, np.newaxis, np.newaxis], axis=0) / np.mean(weights)
            mean_new = np.reshape(mean_new, [self.n_bfs_p,self.dmp.n_dmps])
            mean_old = self.dmp.w
            self.dmp.w = mean_new.transpose()
            self.covar = covar_decay_factor**2 * self.covar
            #start_time = time.time()
            x_track, dx_track, ddx_track, t_track, _, _ = self.dmp.rollout(tau=self.tau)

            policy_cost, label_z[ii], fin = self.evaluate_rollout_one_param_acc(x_track, ddx_track, max_val)

            train_data[ii, :, :] = mean_new
            avg_costs[ii, :] = policy_cost

            if ii % self.plot_frequency == 0 or fin or ii == max_runs:

                new_value_H = [-0.02, -policy_cost[1], '0']

                H.set_text(r"$s_1=$"+str(np.round(-policy_cost[1],4)))
                H.set_position((new_value_H[0], new_value_H[1]))

                f_c = -policy_cost[1]/max_val
                f_c = max(0, min(f_c, 1))

                self.plot_rollout(ax1,[0,0,f_c],1)

                circle_patch.set_radius(-policy_cost[1])
                ax1.set_xlim([-0.02, 0.17])
                ax1.set_ylim([-0.01, max_val+0.02]) #0.22
                ax1.set_title("DMP trajectory at iteration j=" + str(ii))
                ax6.clear()
                    #ax6 = fig2.add_subplot(1, 2, 1)
                ax6.set_xlabel(r"$\theta_{i,\mathbf{e}_1}$")
                ax6.set_ylabel("Activations")
                ax6.set_ylim([-1.0, 3])
                plt.suptitle("Forcing term parameters at iteration j=" + str(ii))

                ax7.clear()
                #ax7 = fig2.add_subplot(1, 2, 2)
                ax7.set_xlabel(r"$\theta_{i,\mathbf{e}_2}$")
                #ax7.set_ylabel("Activations")
                ax7.set_ylim([-1.5, 2.5])

                new_means_2 = self.dmp.w[1, :]
                new_means_3 = self.dmp.w[2, :]

                ba1 = ax6.bar(range(len(new_means_2)), new_means_2, color=[0,0,f_c])
                ba2 = ax7.bar(range(len(new_means_3)), new_means_3, color=[0,0,f_c])
                #fig2.canvas.draw()
                #print("optimization counter: ", ii, "policy cost: ", policy_cost)
                if visualize_progress:
                    plt.pause(0.1)



            if ii == max_runs or fin:

                print("final costs, accelerations, jerk, and first acceleration: ",policy_cost,np.linalg.norm(ddx_track),np.linalg.norm(np.diff(ddx_track,axis=0)),np.sum(np.absolute(ddx_track[0,:])))
                ti_end = time.time()
                ti = ti_end-ti_start
                print(f'finished optimization at iteration {ii} after {ti:2f}s, mean_z_0 = {mean_new[0,2]}')
                break
        plt.ion() #allows the execution of code after the plt.show(), e.g. the closing of plots
        plt.show()
        #fig_handle = plt.gcf()
        return avg_costs, h, train_data, label_z, ii, ti

    def costs_to_weights(self, full_costs, h):
        costs_range = np.max(full_costs) - np.min(full_costs)

        if costs_range == 0:
            weights = np.ones(len(full_costs))
        else:
            weights = np.exp(-h * (full_costs - np.min(full_costs)) / costs_range)
        weights = weights / np.sum(weights)

        return weights

    def evaluate_rollout_2D(self, x_track, ddx_track, z_high, z_low,x_high, max_val_z, min_val_z,max_val_x, weights_z,mode=""):
        #def evaluate_rollout(self, m, z, max_val):
        y = x_track
        y_end = y[-1, :]
        fin = False
        y_borders_z = np.zeros(len(z_high))
        heights_z = np.zeros(len(z_high))
        zi = np.zeros(len(z_high))
        start = False
        #print("len z: ",len(z))
        for iii in range(len(z_high)):
            #y2_z = (iii+1) * self.dmp.x_goal[1] / (len(z) + 1) #possible to mark any of the borders between init and goal in meters
            id = np.argmin(np.abs(y[:, 1] - z_high[iii])) #find the time step that is available in the trajectory that is closest to y2_z
            heights_z[iii] = y[id, 2]
            zi[iii] = y[id, 2]
            y_borders_z[iii] = y[id, 1]

        y_borders_x = np.zeros(len(x_high))
        heights_x = np.zeros(len(z_high))
        xi = np.zeros(len(x_high))
        id_high_x = []
        #print("len z: ",len(z))
        for iii in range(len(x_high)):
            #y2_z = (iii+1) * self.dmp.x_goal[1] / (len(z) + 1) #possible to mark any of the borders between init and goal in meters
            id = np.argmin(np.abs(y[:, 1] - x_high[iii])) #find the time step that is available in the trajectory that is closest to y2_z
            id_high_x.append(id)
            heights_x[iii] = y[id, 0]
            xi[iii] = y[id, 0]
            y_borders_x[iii] = y[id, 1]

        y_borders_z_low = np.zeros(len(z_low))
        lows_z = np.zeros(len(z_low))
        id_low_z = []
        #print("len z: ",len(z))
        for iii in range(len(z_low)):
            #y2_z = (iii+1) * self.dmp.x_goal[1] / (len(z) + 1) #possible to mark any of the borders between init and goal in meters
            id = np.argmin(np.abs(y[:, 1] - z_low[iii])) #find the time step that is available in the trajectory that is closest to y2_z
            id_low_z.append(id)
            lows_z[iii] = y[id, 2]
            y_borders_z_low[iii] = y[id, 1]

        costs = np.zeros(6)

        x_weight = max_val_x/max_val_z
        #print("pi2 2D cost evaluation, max_val_x/max_val_z: ", max_val_x, max_val_z)
        maxs_x = np.minimum(0, max_val_x - y[id_high_x[0]:id_high_x[1], 0] + 0.002)
        mins_x = np.minimum(0, 0.002 + y[id_high_x[0]:id_high_x[1], 0] - max_val_x)
        costs[4] = -np.sum(maxs_x) - np.sum(mins_x)

        costs[1] = -np.min(heights_z*weights_z)  # max(cost_z)
        costs[3] = -np.sum(np.minimum(0, min_val_z - y[id_low_z[0]:id_low_z[1], 2] + 0.002)) - np.sum(
                np.minimum(0, 0.002 + y[id_low_z[0]:id_low_z[1], 2] - min_val_z))
        if costs[4] == 0.0 and costs[3] == 0.0:
        #    costs[1] = -np.min(heights_z*weights_z)  # max(cost_z)
            start = True

        costs[2] = -10 * np.sum(y[:, 2][y[:, 2]<0]) #punish values that collide with the table (z values below 0)
        costs[5] = 0 #1/500*np.sum(np.absolute(ddx_track[0,:])) #+ 1/500*np.linalg.norm(np.diff(ddx_track,axis=0)))
        costs[0] = costs[1] + costs[2] + costs[3] + costs[4] + costs[5]

        min_max_x = np.array([np.min(mins_x),np.max(-maxs_x)])

        if costs[1]<-max_val_z*np.min(weights_z):
            fin = True
        #print(xi)
        return costs, zi, xi, start,fin, heights_z, min_max_x, y_borders_z,y_borders_x

    def evaluate_rollout_continuous_box_acc(self, x_track, ddx_track, z, max_val, mode=""):
        #def evaluate_rollout(self, m, z, max_val):
        y = x_track
        y_end = y[-1, :]
        #yd = m[:, 1 + self.dmp.n_dmps:1 + 2 * self.dmp.n_dmps]
        #ydd = m[:, 1 + 2 * self.dmp.n_dmps:1 + 3 * self.dmp.n_dmps]
        fin = False
        y_borders = np.zeros(len(z))
        heights_z = np.zeros(len(z))
        lows_z = np.zeros(len(z))
        zi = np.zeros(len(z))
        #print("len z: ",len(z))
        costs = np.zeros(5)
        cost_concave = 0
        if len(z)==2:
            id0 = np.argmin(np.abs(y[:, 1] - z[0]))
            id1 = np.argmin(np.abs(y[:, 1] - z[1]))
            if id0 == id1: #avoid empty arrays
                id0-=1
            zi[0] = y[id0, 2]
            zi[1] = y[id1, 2]
            y_borders[0] = y[id0, 1]
            y_borders[1] = y[id1, 1]
            y_inits = y[:id0,1]
            y_ends = y[id1:,1]

            costs[1] = -np.min(y[id0:id1, 2])  # max(cost_z)
        else:
            for iii in range(len(z)):
                #y2_z = (iii+1) * self.dmp.x_goal[1] / (len(z) + 1) #possible to mark any of the borders between init and goal in meters
                id = np.argmin(np.abs(y[:, 1] - z[iii])) #find the time step that is available in the trajectory that is closest to y2_z
                heights_z[iii] = y[id, 2]
                zi[iii] = y[id, 2]
                y_borders[iii] = y[id, 1]



            costs[1] = -np.min(heights_z)  # max(cost_z)

        costs[3] = -np.sum(np.minimum(0, self.bounds[0] + y[:, 1] - self.dmp.x_0[1])) - np.sum(
                np.minimum(0, self.bounds[1] + self.dmp.x_goal[1] - y[:, 1]))

        costs[2] = -10 * np.sum(y[:, 2][y[:, 2]<0]) #+ 0.1*concave_cost#+ 0.1*np.sum(np.diff(y[:,1]))#+ cost_concave #punish values that collide with the table (z values below 0) and concave traj
        costs[4] = 1/100*np.sum(np.absolute(ddx_track[0,:])) + 1/20*np.linalg.norm(np.diff(ddx_track,axis=0))
        #costs[4] = 1/20*np.sum(np.absolute(ddx_track[0,:])) + 1/40*np.linalg.norm(np.diff(ddx_track))
        costs[0] = costs[1] + costs[2] + costs[3] + costs[4]

        min_max_x_z = np.array([np.min(y[:,1]),np.max(y[:,1]),np.min(y[:,2]),np.max(y[:,2])])

        if costs[1] < -max_val:
            fin = True

        return costs, zi, fin, y_borders, min_max_x_z

    def evaluate_rollout_one_param_acc(self, x_track, ddx_track, max_val):
        #def evaluate_rollout(self, m, z, max_val):
        y = x_track
        y_end = y[-1, :]
        #yd = m[:, 1 + self.dmp.n_dmps:1 + 2 * self.dmp.n_dmps]
        #ydd = m[:, 1 + 2 * self.dmp.n_dmps:1 + 3 * self.dmp.n_dmps]
        fin = False
        zi = 0
        #print("len z: ",len(z))
        costs = np.zeros(5)
        #y_ = y[(y[:, 1] >= y[0,1]) & (y[:, 1] <= y[-1,1])] #moves along y-axis
        distances = distance.cdist([y_end[1:3]/2], y[:,1:3])
        radius = np.min(distances)


        zi=radius
        costs[1] = -radius  # max(cost_z)

        #no costs for bounds to left/right
        costs[3] = 0 #-np.sum(np.minimum(0, self.bounds[0] + y[:, 1] - self.dmp.x_0[1])) - np.sum(
                #np.minimum(0, self.bounds[1] + self.dmp.x_goal[1] - y[:, 1]))

        costs[2] = -10 * np.sum(y[:, 2][y[:, 2]<0]) #punish values that collide with the table (z values below 0)
        costs[4] = 1/100*np.sum(np.absolute(ddx_track[0,:])) + 1/200*np.linalg.norm(np.diff(ddx_track,axis=0))
        costs[0] = costs[1] + costs[2] + costs[3] + costs[4]


        if costs[1] < -max_val:
            fin = True

        return costs, zi, fin

    def exponential_distribution(self,n, start, end, steepness):
        # Step 1: Generate a linear space of exponents
        exponents = np.linspace(0, 1, num=n)
        # Step 2: Apply steepness factor to the exponents (modify growth)
        exponents = exponents ** steepness  # steeper growth when steepness > 1
        # Step 3: Map the exponent range to the start and end range
        exp_array = start + (end - start) * exponents  # scale to the desired start and end
        return exp_array

    def get_covar(self):
        sigmas = np.linspace(self.sigma_min, self.sigma_max, self.n_bfs_p)
        sigmas = self.exponential_distribution(self.n_bfs_p,self.sigma_min,self.sigma_max,self.sigma_steepness)
        covar_dmp = (np.exp(sigmas)-1)**2 * np.eye(self.n_bfs_p) #[:, :, np.newaxis]-1) * np.ones((self.dmp.n_bfs, self.dmp.n_bfs, self.dmp.n_dmps))
        self.covar = np.stack([covar_dmp] * self.dmp.n_dmps, axis=-1)
        #print("init covar: ", self.covar)
    def explore(self, n_samples):
        mean_expl = np.zeros((n_samples, self.n_bfs_p, self.dmp.n_dmps))
        #print("explore covar: ", self.covar)
        for j in range(self.dmp.n_dmps):
            if self.param_pertub[j]:
                mean_expl[:, :, j] = np.random.multivariate_normal(self.dmp.w[j, :],self.covar[:, :, j], n_samples)
            else:
                mean_expl[:, :, j] = np.zeros((n_samples, self.n_bfs_p)) + self.dmp.w[j, :]

        return mean_expl

    def plot_rollout(self, plot_ax, col_ball="c", lineW_traj=0.3):
        x_track, _, _, _, _, _ = self.dmp.rollout(tau=self.tau)
        #y = m[:, 1:1 + self.dmp.n_dmps]
        plot_ax.plot(x_track[:, 1], x_track[:, 2], color=col_ball, linewidth=lineW_traj)
        plot_ax.scatter(self.dmp.x_goal[1], self.dmp.x_goal[2], marker='o', color=col_ball)
        plot_ax.scatter(self.dmp.x_0[1], self.dmp.x_0[2], marker='o', color=col_ball)

    def plot_rollout_x(self, plot_ax, col_ball="c", lineW_traj=0.3):
        x_track, _, _, _, _, _ = self.dmp.rollout(tau=self.tau)
        #y = m[:, 1:1 + self.dmp.n_dmps]
        plot_ax.plot(x_track[:, 1], x_track[:, 0], color=col_ball, linewidth=lineW_traj)
        plot_ax.scatter(self.dmp.x_goal[1], self.dmp.x_goal[0], marker='o', color=col_ball)
        plot_ax.scatter(self.dmp.x_0[1], self.dmp.x_0[0], marker='o', color=col_ball)

    def plot_rollout_3d(self, plot_ax, col_ball="c", lineW_traj=0.3):
        x_track, _, _, _, _, _ = self.dmp.rollout(tau=self.tau)
        #y = m[:, 1:1 + self.dmp.n_dmps]
        # Plot the trajectory
        plot_ax.plot(x_track[:, 0], x_track[:, 1], x_track[:, 2], color=col_ball, linewidth=lineW_traj)
        # Plot the start and goal points
        plot_ax.scatter(self.dmp.x_0[0], self.dmp.x_0[1], self.dmp.x_0[2], marker='o', color=col_ball)
        plot_ax.scatter(self.dmp.x_goal[0], self.dmp.x_goal[1], self.dmp.x_goal[2], marker='o', color=col_ball)
