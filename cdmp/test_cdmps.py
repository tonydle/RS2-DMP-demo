#!/usr/bin/python3

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.cdmps import CartesianDMPs


if __name__ == '__main__':

    # # --- PATAMETERS --- # # 

    # CDMPs
    alpha_s_val = 7.0
    k_gain_val   = 1000.0
    rbfs_pSec_val = 4.0

    # Temporal Scaling
    tau_scaler_val = 1.0
    
    # Spatial Scaling
    rep_relpose_start = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) # [x,y,z, qw,qx,qy,qz]    
    rep_relpose_goal  = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) # [x,y,z, qw,qx,qy,qz]
    
    # Obstacle Avoidance
    obstacle_avoidance = True
    beta_val = 5.0
    lambda_f_val = 5.0
    eta_val = 5.0
    obstacle_pos = np.array([0.6, 0.2, 0.2])
    plot_3dtraj = True
    plot_forces = False


    # # --- DEMO DATA --- # #
    
    filename = 'minJerk1.txt'
    filepath = os.path.join(os.path.dirname(__file__), 'demos', filename)
    if not os.path.exists(filepath):
        sys.exit('ERROR: Unable to locate file: {}'.format(filepath))

    data = np.loadtxt(filepath, delimiter=',', skiprows=1)

    with open(filepath, 'r') as file:
        column_names = file.readline().rstrip().split(',')

    dem_time   = data[:, column_names.index("TimeStep")]
    dem_pos    = data[:, [column_names.index("PosX"), column_names.index("PosY"), column_names.index("PosZ")]]
    dem_quat   = data[:, [column_names.index("QuatW"), column_names.index("QuatX"), column_names.index("QuatY"), column_names.index("QuatZ")]]


    # # --- LEARN CDMPs --- # # 

    test_cdmp = CartesianDMPs()

    test_cdmp.load_demo(filename   = filename,
                        dem_time   = dem_time,
                        dem_pos    = dem_pos,
                        dem_quat   = dem_quat)

    test_cdmp.learn_cdmp(alpha_s  = alpha_s_val,
                         k_gain    = k_gain_val,
                         rbfs_pSec = rbfs_pSec_val)
    

    # # --- REPRODUCE CDMPs --- # #

    test_cdmp.init_reproduction(tau_scaler = tau_scaler_val, rep_relpose_start=rep_relpose_start, rep_relpose_goal=rep_relpose_goal)
    
    rep_force = np.zeros(3)
    rep_forces = []

    for t in range(len(test_cdmp.rep_cs.s_track)):

        curr_pos, curr_linVel, curr_quat = test_cdmp.rollout_step(curr_s_step=test_cdmp.rep_cs.s_track[t], ext_force=rep_force[0])

        # --- Obstacle Avoidance --- #
        if obstacle_avoidance: 
            d = np.linalg.norm(curr_pos - obstacle_pos)
            nabla_d = (curr_pos - obstacle_pos) / d
            nabla_d_norm = np.linalg.norm(nabla_d)

            curr_linVel_norm = np.linalg.norm(curr_linVel)

            cos_theta = np.dot(curr_linVel[0], nabla_d[0]) / (curr_linVel_norm * nabla_d_norm)

            if math.pi / 2 < math.acos(cos_theta) <= math.pi:
                rep_force = ((-cos_theta) ** beta_val * lambda_f_val * eta_val * curr_linVel_norm * nabla_d) / (d ** (eta_val + 1))
            else:
                rep_force = np.zeros(3)
            
            rep_forces.append(rep_force)
            

    # # --- PLOTTING --- # #

    # --- CDMPs inherent plots --- #
    # test_cdmp.analysis_plots(feature='wi')
    # test_cdmp.analysis_plots(feature='phases')
    # test_cdmp.analysis_plots(feature='psis')
    # test_cdmp.analysis_plots(feature='forces')
    # test_cdmp.analysis_plots(feature='positions')
    # test_cdmp.analysis_plots(feature='quaternions')
    # test_cdmp.analysis_plots(feature='poses')


    # 3D Plot of trajectories including obstacle
    if plot_3dtraj:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(test_cdmp.dem_pos[:, 0], test_cdmp.dem_pos[:, 1], test_cdmp.dem_pos[:, 2], label='dem')
        ax.plot(test_cdmp.rep_pos[:, 0], test_cdmp.rep_pos[:, 1], test_cdmp.rep_pos[:, 2], label='rep')
        if obstacle_pos is not None:
            ax.scatter(obstacle_pos[0], obstacle_pos[1], obstacle_pos[2], color='r', label='obstacle')
        ax.scatter(test_cdmp.rep_Pstart[0], test_cdmp.rep_Pstart[1], test_cdmp.rep_Pstart[2], color='g', label='rep start')
        ax.scatter(test_cdmp.dem_Pstart[0], test_cdmp.dem_Pstart[1], test_cdmp.dem_Pstart[2], color='y', label='dem start')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Trajectory')
        ax.legend()
        plt.show()

    # Plot rep_forces
    if plot_forces:
        rep_forces = np.vstack(rep_forces)

        fig, ax = plt.subplots()
        ax.plot(rep_forces[:, 0], label='Force X')
        ax.plot(rep_forces[:, 1], label='Force Y')
        ax.plot(rep_forces[:, 2], label='Force Z')

        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Force Magnitude')
        ax.set_title('External Forces on CDMPs from Potential Field')
        ax.legend()
        plt.show()
