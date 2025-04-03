#!/usr/bin/python3

import os
import sys
import numpy as np

from src.cdmps import CartesianDMPs


if __name__ == '__main__':

    # # --- PATAMETERS --- # # 

    # CDMPs
    alpha_s_val = 7.0
    k_gain_val   = 50.0
    rbfs_pSec_val = 4.0

    # Temporal Scaling
    tau_scaler_val = 1.0
    
    # Spatial Scaling
    rep_relpose_start = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) # [x,y,z, qw,qx,qy,qz]    
    rep_relpose_goal  = np.array([-0.3, 0.2, 0.1, 1.0, 0.0, 0.0, 0.0]) # [x,y,z, qw,qx,qy,qz]


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
    
    for t in range(len(test_cdmp.rep_cs.s_track)):

        next_pos,next_quat = test_cdmp.rollout_step(curr_s_step = test_cdmp.rep_cs.s_track[t])


    # # --- PLOTTING --- # #

    # test_cdmp.analysis_plots(feature='wi')
    # test_cdmp.analysis_plots(feature='phases')
    # test_cdmp.analysis_plots(feature='psis')
    # test_cdmp.analysis_plots(feature='forces')
    # test_cdmp.analysis_plots(feature='positions')
    # test_cdmp.analysis_plots(feature='quaternions')
    test_cdmp.analysis_plots(feature='poses')

     




