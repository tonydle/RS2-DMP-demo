'''
Copyright (C) 2023 Victor Hernandez Moreno
Copyright (C) 2020 Michele Ginesi

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
'''

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyquaternion import Quaternion


# ---------------------------------------------------------------------------- #
# DMPs in Cartesian Space
# ---------------------------------------------------------------------------- #

class CartesianDMPs(object):
    '''
    Implementation of discrete Dynamic Movement Primitives in cartesian space,
    as described in
    [1] 

    '''

    def __init__(self, **kwargs):

        self.alpha_s    = None
        self.K          = None
        self.D          = None
        self.rbfs_pSec  = None
        self.num_rbfs   = None
        self.tau_scaler = None
        self.rep_tau    = None

        self.dem_fp     = np.zeros(3)
        self.wp         = np.zeros(3)
        self.wo         = np.zeros(3)
        self.rep_fp     = np.zeros(3)

        self.reset_repstates()


    def reset_repstates(self):

        self.rep_time   = np.zeros(1)
        self.rep_pos    = np.zeros(3)
        self.rep_linVel = np.zeros(3)
        self.rep_linAcc = np.zeros(3)
        self.rep_quat   = np.zeros(4)
        self.rep_eq     = np.zeros(3)
        self.rep_edq    = np.zeros(3)
        self.rep_eddq   = np.zeros(3)

        self.rep_Pstart = np.zeros(3)
        self.rep_Qstart = Quaternion(1.0, 0.0, 0.0, 0.0)
        self.rep_Pgoal  = np.zeros(3)
        self.rep_Qgoal  = Quaternion(1.0, 0.0, 0.0, 0)


    def disp_params(self):

        if not np.allclose(self.rep_Pstart, np.zeros(3)):
            Pstart_offset = self.rep_Pstart - self.dem_Pstart
            Pgoal_offset  = self.rep_Pgoal - self.dem_Pgoal
        else: 
            Pstart_offset = None
            Pgoal_offset  = None

        if self.rep_Qstart != Quaternion(1.0,0.0,0.0,0.0):
            Qstart_offset = self.quat_diff_euler(self.rep_Qstart, self.dem_Qstart)
            Qgoal_offset  = self.quat_diff_euler(self.rep_Qgoal, self.dem_Qgoal)
        else: 
            Qstart_offset = None
            Qgoal_offset  = None

        print('\n filename:         \t %s' % self.filename , 
              '\n dem duration:     \t %s' % self.dem_tau ,
              '\n dt:               \t %s' % self.dt , 
              '\n dims:             \t %s' % self.dims ,
              '\n alpha:            \t %s' % self.alpha_s ,
              '\n K:                \t %s' % self.K , 
              '\n D:                \t %s' % self.D , 
              '\n rbfs_pSec:        \t %s' % self.rbfs_pSec ,
              '\n num_rbfs:         \t %s' % self.num_rbfs , 
              '\n tau_scaler:       \t %s' % self.tau_scaler , 
              '\n rep duration:     \t %s' % self.rep_tau , 
              '\n',
              '\n dem_pos size:     \t %s' % self.dem_pos.shape[0] ,
              '\n dem_quat size:    \t %s' % self.dem_quat.shape[0] ,
              '\n rep_pos size:     \t %s' % self.rep_pos.shape[0] ,
              '\n rep_quat size:    \t %s' % self.rep_quat.shape[0] ,
              '\n',
              '\n Pstart_offset:    \t %s' % Pstart_offset , 
              '\n Pgoal_offset:     \t %s' % Pgoal_offset ,
              '\n Qstart_offset:    \t %s' % Qstart_offset ,
              '\n Qgoal_offset:     \t %s' % Qgoal_offset , 
              '\n \n')
        

    def load_demo(self, filename, dem_time, dem_pos, dem_quat = None, **kwargs): #, dem_gripper = None

        self.filename   = filename
        self.dem_time   = dem_time.copy()
        self.dem_tau    = self.dem_time[-1]
        self.dt         = self.dem_time[2] - self.dem_time[1]
        self.dims       = dem_pos.shape[1]
        
        self.dem_pos    = dem_pos.copy()     # dem_pos size: [timesteps x 3]
        self.dem_Pstart = dem_pos[0].copy()
        self.dem_Pgoal  = dem_pos[-1].copy()

        if dem_quat is None:
            self.dem_quat   =   np.tile([1.0, 0.0, 0.0, 0.0], [self.dem_pos.shape[0], 1])
            self.dem_Qstart = Quaternion(1.0, 0.0, 0.0, 0.0)
            self.dem_Qgoal  = Quaternion(1.0, 0.0, 0.0, 0.0)
        else:
            self.dem_quat   = dem_quat.copy()
            self.dem_Qstart = Quaternion(dem_quat[0].copy())
            self.dem_Qgoal  = Quaternion(dem_quat[-1].copy())

        # if dem_gripper is not None:
        #     self.dem_gripper = dem_gripper.copy()
        # else:
        #     self.dem_gripper = None

        print('\nLoading demo data completed\n')
        self.disp_params()


    def learn_cdmp(self, alpha_s, k_gain, rbfs_pSec, **kwargs):

        self.alpha_s    = copy.deepcopy(alpha_s)
        self.K          = copy.deepcopy(k_gain)
        self.D          = 2.0 * np.sqrt(self.K)
        self.rbfs_pSec  = copy.deepcopy(rbfs_pSec)

        self.num_rbfs   = np.max([10, int(np.round(self.rbfs_pSec*self.dem_tau))])

        self.dem_cs = CanonicalSystem(dt = self.dt, 
                                      alpha_s = self.alpha_s, 
                                      num_rbfs = self.num_rbfs, 
                                      tau = self.dem_tau,
                                      size = self.dem_pos.shape[0])
            
        sum_psi = np.sum(self.dem_cs.psi_track, 0)
        
        ''' --------------- TRANSLATION --------------- '''

        # Alternative for vel/acc calculation !
        dem_pos_df    = pd.DataFrame(self.dem_pos)
        dem_linVel_df = dem_pos_df.diff().fillna(0.0) / self.dt
        dem_linAcc_df = dem_linVel_df.diff().fillna(0.0) / self.dt

        self.dem_linVel = dem_linVel_df.to_numpy()
        self.dem_linAcc = dem_linAcc_df.to_numpy()

        ## Find the force required to move along this trajectory
        dem_fp = ( self.dem_tau * self.dem_tau * self.dem_linAcc.T / self.K 
                  + self.D * self.dem_tau * self.dem_linVel.T / self.K
                  + (self.dem_pos.T - np.tile(np.reshape(self.dem_Pgoal, [self.dims, 1]), (1,self.dem_pos.T.shape[1])))  
                  + np.reshape((self.dem_Pgoal - self.dem_Pstart), [self.dims, 1]) * self.dem_cs.s_track)

        # Compute useful quantities
        P = self.dem_cs.psi_track / sum_psi * self.dem_cs.s_track

        # Compute the weights using linear regression
        self.wp = np.nan_to_num(dem_fp @ np.linalg.pinv(P))

        self.dem_fp = dem_fp.transpose()

        ''' --------------- ROTATION --------------- '''

        # Do = quatlog(quatnormalize(quatmultiply(tsdmp.demo_quat(end,:),quatconj(tsdmp.demo_quat(1,:)))));
        # Do = self.multlogQuats(self.dem_Qgoal, self.dem_Qstart)

        # demo_e_q = 2 * quatlog(quatnormalize(quatmultiply(tsdmp.demo_quat(end,:),quatconj(tsdmp.demo_quat))));
        dem_eq = np.zeros([self.dem_pos.T.shape[1],3])
        
        for t in range(self.dem_pos.T.shape[1]):
            curr_quat = Quaternion(self.dem_quat[t, :])
            curr_eq = 2 * self.logmultconjQuats(self.dem_Qgoal, curr_quat)
            dem_eq[t, :] = np.reshape(curr_eq.imaginary, [1, 3])
        
        self.dem_eq = dem_eq
        dem_eq_df   = pd.DataFrame(dem_eq)
        dem_edq_df  = dem_eq_df.diff().fillna(0.0) / self.dt
        dem_eddq_df = dem_edq_df.diff().fillna(0.0) / self.dt

        self.dem_edq = dem_edq_df.to_numpy()
        self.dem_eddq = dem_eddq_df.to_numpy()
        
        # tsdmp.demo_fo =  1/tsdmp.K * ( tsdmp.demo_tau^2 * demo_edd_q ...
        #        + tsdmp.D * tsdmp.demo_tau * demo_ed_q ...
        #        + tsdmp.K * demo_e_q(:,2:4) ...
        #        + tsdmp.K * 2 .* Do(2:4).*tsdmp.demo_s );
        dem_fo = ( self.dem_tau * self.dem_tau * self.dem_eddq.T / self.K
                  + self.D * self.dem_tau * self.dem_edq.T / self.K
                  + self.dem_eq.T
                  + 2.0 * np.reshape(self.logmultconjQuats(self.dem_Qgoal, self.dem_Qstart).imaginary, [3,1]) * self.dem_cs.s_track)

        # Compute the weights using linear regression
        self.wo = np.nan_to_num(dem_fo @ np.linalg.pinv(P))

        self.dem_fo = dem_fo.transpose()

        print('\nLearning CDMP completed\n')
        self.disp_params()

        
    def init_reproduction(self, tau_scaler, rep_relpose_start = None, rep_relpose_goal = None, rep_abspose_start = None, **kwargs):

        self.tau_scaler = copy.deepcopy(tau_scaler)
        self.rep_tau    = self.tau_scaler * self.dem_tau

        if rep_relpose_start is not None: 
            # print('\n rep_relpose_start is None ! \n')
            dem_Pstart = self.dem_Pstart
            dem_Qstart = self.dem_Qstart

            rel_Pstart = rep_relpose_start[0:3]
            rel_Qstart = Quaternion(rep_relpose_start[3:7])

            self.rep_Pstart = dem_Pstart + rel_Pstart
            self.rep_Qstart = rel_Qstart * dem_Qstart

        elif rep_abspose_start is not None:
            print('rep_abspose_start SELECTED')
            self.rep_Pstart = rep_abspose_start[0:3]
            self.rep_Qstart = Quaternion(rep_abspose_start[3:7])
        
        else:
            self.rep_Pstart = self.dem_Pstart
            self.rep_Qstart = self.dem_Qstart

        if rep_relpose_goal is None:
            rep_Pgoal = self.dem_Pgoal
            rep_Qgoal  = self.dem_Qgoal
        else:
            dem_Pgoal = self.dem_Pgoal
            dem_Qgoal = self.dem_Qgoal

            rel_Pgoal = rep_relpose_goal[0:3]
            rel_Qgoal = Quaternion(rep_relpose_goal[3:7])

            rep_Pgoal = dem_Pgoal + rel_Pgoal
            rep_Qgoal = rel_Qgoal * dem_Qgoal

        self.rep_Pgoal  = copy.deepcopy(rep_Pgoal)
        self.rep_Qgoal  = copy.deepcopy(rep_Qgoal)

        self.rep_cs = CanonicalSystem(dt = self.dt, 
                                      alpha_s = self.dem_cs.alpha_s, 
                                      num_rbfs = self.dem_cs.num_rbfs, 
                                      tau = self.rep_tau,
                                      size = int(np.round(self.rep_tau / self.dt)))

        self.rep_time   = np.zeros([1,1])
        self.rep_s_step = np.ones([1,1])
        self.rep_pos    = np.array([self.rep_Pstart])  # ' rep is size 1x3 : (x,y,z)'
        self.rep_linVel = np.zeros([1,self.dims])
        self.rep_linAcc = np.zeros([1,self.dims])
        self.rep_fp     = np.zeros([1,self.dims])

        self.rep_quat   = np.array([self.rep_Qstart.elements])
        self.rep_eq     = np.array([2.0 * self.logmultconjQuats(self.rep_Qgoal, self.rep_Qstart).imaginary])
        self.rep_edq    = np.zeros([1,3])
        self.rep_eddq   = np.zeros([1,3])
        self.rep_fo     = np.zeros([1,self.dims])

        self.rep_cs.reset_states()
        self.s_stepnum = 0

        print('\nReproduction initialized\n')
        self.disp_params()
        


    def rollout_step(self, curr_s_step = 1, ext_force = None, ext_torque = None, ext_linVel = None, ext_angVel = None, **kwargs):

        # print('step: \t %s' % self.s_stepnum)

        if ext_force is None:
            ext_force = np.zeros(3)
        if ext_torque is None:
            ext_torque = np.zeros(3)

        if ext_linVel is None:
            ext_linVel = np.zeros(3)
        if ext_angVel is None:
            ext_angVel = np.zeros(3)

        curr_pos    = self.rep_pos[-1, :]
        curr_linVel = self.rep_linVel[-1, :]
        curr_eq     = self.rep_eq[-1, :]
        curr_edq    = self.rep_edq[-1, :] 

        psi_step = self.rep_cs.rbfs_step(s_step = curr_s_step)

        ''' --------------- TRANSLATION --------------- '''

        # fp = forceWp' * psi .* currS;
        next_fp = np.dot(self.wp, psi_step) / np.sum(psi_step) * curr_s_step

        # linAcc =  kPos*(goalPos-currState.pos) / tauPos^2 ...
        #         - dPos*currState.linVel        / tauPos   ...
        #         - kPos*Dp.*currS               / tauPos^2 ...
        #         + kPos*fp                      / tauPos^2;
        next_linAcc = (- self.K * (curr_pos - self.rep_Pgoal) / self.rep_tau / self.rep_tau 
                       - self.D * curr_linVel / self.rep_tau
                       - self.K * (self.rep_Pgoal - self.rep_Pstart) * curr_s_step / self.rep_tau / self.rep_tau
                       + self.K * next_fp / self.rep_tau / self.rep_tau
                       + ext_force.T / self.rep_tau / self.rep_tau)
        
        next_linVel = curr_linVel + next_linAcc * self.dt + ext_linVel.reshape(1, self.dims) * self.dt
        next_pos    = curr_pos    + next_linVel * self.dt

        next_pos    = next_pos.reshape(1, self.dims)
        next_linVel = next_linVel.reshape(1, self.dims)
        next_linAcc = next_linAcc.reshape(1, self.dims)
        next_fp     = next_fp.reshape(1, self.dims)

        ''' --------------- ROTATION --------------- '''

        # fo = forceWo' * psi .* currS;
        next_fo = np.dot(self.wo, psi_step) / np.sum(psi_step) * curr_s_step

        # edd_q = - kPos * currState.e_q    / tauPos^2  ...
        #         - dPos * currState.ed_q   / tauPos    ...
        #         - kPos*2.*Do(2:4)'.*currS / tauPos^2  ...
        #         + kPos*fo                 / tauPos^2  ;
        next_eddq = (- self.K * curr_eq / self.rep_tau / self.rep_tau 
                     - self.D * curr_edq / self.rep_tau
                     - self.K * 2.0 * self.logmultconjQuats(self.rep_Qgoal, self.rep_Qstart).imaginary * curr_s_step / self.rep_tau / self.rep_tau
                     + self.K * next_fo / self.rep_tau / self.rep_tau
                     - ext_torque.T / self.rep_tau / self.rep_tau)
        
        next_edq = curr_edq + next_eddq * self.dt - ext_angVel.reshape(1, 3) * self.dt
        next_eq  = curr_eq  + next_edq  * self.dt

        # quat = quatmultiply( quatconj( quatexp(1/2 * [0 e_q' ])) , goalQuat' );
        quat      = Quaternion(1/2 * np.insert(next_eq, 0, 0.0))
        quatexp   = Quaternion.exp(quat)
        quatconj  = quatexp.conjugate
        next_quat = quatconj * self.rep_Qgoal

        next_quat = next_quat.elements.reshape(1, 4)
        next_eq   = next_eq.reshape(1, 3)
        next_edq  = next_edq.reshape(1, 3)
        next_eddq = next_eddq.reshape(1, 3)
        next_fo   = next_fo.reshape(1, 3)

        ''' --------------- SAFE REPRODUCTION STEP --------------- '''

        self.rep_time   = np.append(self.rep_time, self.rep_time[-1] + self.rep_cs.dt)
        self.rep_s_step = np.append(self.rep_s_step, curr_s_step)

        self.rep_pos    = np.append(self.rep_pos, next_pos, axis=0)
        self.rep_linVel = np.append(self.rep_linVel,next_linVel, axis=0)
        self.rep_linAcc = np.append(self.rep_linAcc,next_linAcc, axis=0)
        self.rep_fp     = np.append(self.rep_fp,next_fp, axis=0)
        
        self.rep_quat   = np.append(self.rep_quat, next_quat, axis=0)
        self.rep_eq     = np.append(self.rep_eq,next_eq, axis=0)
        self.rep_edq    = np.append(self.rep_edq,next_edq, axis=0)
        self.rep_eddq   = np.append(self.rep_eddq,next_eddq, axis=0)
        self.rep_fo     = np.append(self.rep_fo,next_fo, axis=0)

        self.s_stepnum += 1

        return next_pos, next_linVel, next_quat


    def logmultconjQuats(self, quat1, quat2):

        multiplied_quat = quat1 * quat2.conjugate
        log_quat = Quaternion.log(multiplied_quat.normalised)

        return log_quat
    

    def quat_diff_euler(self, q1, q2):

        # Convert quaternions to rotation matrices
        R1 = q1.rotation_matrix
        R2 = q2.rotation_matrix
        
        # Calculate the relative rotation matrix
        R_rel = np.matmul(R2, np.linalg.inv(R1))
        
        # Extract the elements of the relative rotation matrix
        r11, r12, r13 = R_rel[0, :]
        r21, r22, r23 = R_rel[1, :]
        r31, r32, r33 = R_rel[2, :]
        
        # Calculate the Euler angles from the relative rotation matrix
        roll = np.arctan2(r32, r33)
        pitch = np.arctan2(-r31, np.sqrt(r32**2 + r33**2))
        yaw = np.arctan2(r21, r11)
        
        # Convert the Euler angles to degrees
        euler_angles = np.array([np.degrees(roll), np.degrees(pitch), np.degrees(yaw)])
        
        return euler_angles
    

    def analysis_plots(self, feature = None, obstacle_pos = None, **kwargs):

        if feature == 'phases':
            plt.figure
            plt.plot(self.dem_cs.s_track)
            plt.plot(self.rep_cs.s_track)
            plt.xlabel('Time Index')
            plt.ylabel('phase s')
            plt.title('Canonical System')
            plt.legend(['dem', 'rep'])
            plt.show()

        elif feature == 'psis':
            _, axs = plt.subplots(2, 1)
            axs[0].plot(self.dem_cs.psi_track.transpose())
            axs[0].set_title('dem psis')
            axs[1].plot(self.rep_cs.psi_track.transpose())
            axs[1].set_title('rep psis')
            plt.legend(['dem', 'rep'])
            plt.show()

        elif feature == 'wi':
            _, axs = plt.subplots(2, 1)
            axs[0].plot(self.wp.transpose())
            axs[0].set_title('wp')
            axs[0].legend(['wp'])
            axs[1].plot(self.wo.transpose())
            axs[1].set_title('wo')
            axs[1].legend(['wo'])
            plt.show()

        elif feature == 'forces':
            _, axs = plt.subplots(2, 2)
            axs[0,0].plot(self.dem_fp)
            axs[0,0].set_title('dem Fp')
            axs[0,1].plot(self.rep_fp)
            axs[0,1].set_title('rep Fp')
            axs[1,0].plot(self.dem_fo)
            axs[1,0].set_title('dem Fo')
            axs[1,1].plot(self.rep_fo)
            axs[1,1].set_title('rep Fo')
            plt.show()
            
        elif feature == 'positions':
            _, axs = plt.subplots(self.dims)
            for i in range(self.dims):
                axs[i].plot(self.dem_time, self.dem_pos[:, i])
                axs[i].plot(self.rep_time, self.rep_pos[:, i])
                axs[i].set_title('Pos {}'.format(i))
                axs[i].legend(['dem', 'rep'])
            plt.show()

        elif feature == 'quaternions':
            _, axs = plt.subplots(2, 2)
            axs[0, 0].plot(self.dem_time, self.dem_quat[:, 0])
            axs[0, 0].plot(self.rep_time, self.rep_quat[:, 0])
            axs[0, 0].set_title('Quat w')
            axs[0, 1].plot(self.dem_time, self.dem_quat[:, 1])
            axs[0, 1].plot(self.rep_time, self.rep_quat[:, 1])
            axs[0, 1].set_title('Quat x')
            axs[1, 0].plot(self.dem_time, self.dem_quat[:, 2])
            axs[1, 0].plot(self.rep_time, self.rep_quat[:, 2])
            axs[1, 0].set_title('Quat y')
            axs[1, 1].plot(self.dem_time, self.dem_quat[:, 3])
            axs[1, 1].plot(self.rep_time, self.rep_quat[:, 3])
            axs[1, 1].set_title('Quat z')
            plt.legend(['dem', 'rep'])
            plt.show()

        elif feature == 'poses':
            fig, axs = plt.subplots(1, 2)
            axs[0].plot(self.dem_time, self.dem_pos[:, 0], label='dem_x')
            axs[0].plot(self.dem_time, self.dem_pos[:, 1], label='dem_y')
            axs[0].plot(self.dem_time, self.dem_pos[:, 2], label='dem_z')
            axs[0].plot(self.rep_time, self.rep_pos[:, 0], label='rep_x')
            axs[0].plot(self.rep_time, self.rep_pos[:, 1], label='rep_y')
            axs[0].plot(self.rep_time, self.rep_pos[:, 2], label='rep_z')
            axs[0].set_title('Position')
            axs[0].legend()
            axs[1].plot(self.dem_time, self.dem_quat[:, 0], label='dem_w')
            axs[1].plot(self.dem_time, self.dem_quat[:, 1], label='dem_x')
            axs[1].plot(self.dem_time, self.dem_quat[:, 2], label='dem_y')
            axs[1].plot(self.dem_time, self.dem_quat[:, 3], label='dem_z')
            axs[1].plot(self.rep_time, self.rep_quat[:, 0], label='rep_w')
            axs[1].plot(self.rep_time, self.rep_quat[:, 1], label='rep_x')
            axs[1].plot(self.rep_time, self.rep_quat[:, 2], label='rep_y')
            axs[1].plot(self.rep_time, self.rep_quat[:, 3], label='rep_z')
            axs[1].set_title('Quaternion')
            axs[1].legend()
            
            plt.show()

        else:
            print('please provide variable name')










'''
Copyright (C) 2023 Victor Hernandez Moreno
Copyright (C) 2018 Michele Ginesi
Copyright (C) 2018 Daniele Meli
Copyright (C) 2013 Travis DeWolf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
'''


class CanonicalSystem():
    '''
    Implementation of the canonical dynamical system
    '''

    def __init__(self, dt, alpha_s, num_rbfs, tau, size, **kwargs):
        
        self.alpha_s    = alpha_s
        self.dt         = dt
        self.num_rbfs   = num_rbfs
        self.tau        = tau
        self.size       = size

        # self.timesteps = int(self.tau / self.dt) + 1 # int( -(-self.tau // self.dt) ) + 1

        self.reset_states()
        self.phase_rollout()
        self.rbfs_rollout()


    def reset_states(self):

        self.s_step = 1.0
        self.psi_step = 0.0


    def phase_rollout(self):
        
        self.reset_states()
        s_track = self.s_step
        loop_count = 0

        while (loop_count < self.size-1):
            # for t in np.linspace(1, self.size-1, self.size): #int(self.tau/self.dt), int(self.tau/self.dt), endpoint=True): # np.arange(self.dt, self.tau, self.dt):
            self.phase_step(self.s_step)
            s_track = np.append(s_track, self.s_step)
            loop_count += 1

        self.s_track = s_track

        return self.s_track


    def phase_step(self, s_step):

        # Since the canonical system is linear, we use an exponential method
        const = - self.alpha_s / self.tau
        s_step *= np.exp(const * self.dt)
        self.s_step = s_step

        return self.s_step


    def rbfs_rollout(self):

        self.rbfs_ci = np.exp( -self.alpha_s / self.tau * np.linspace(0, self.tau, self.num_rbfs))

        # Set the "widths" for the basis functions
        self.rbfs_hi = 1.0 / np.diff(self.rbfs_ci) / np.diff(self.rbfs_ci)
        self.rbfs_hi = np.append(self.rbfs_hi, self.rbfs_hi[-1])

        c = np.reshape(self.rbfs_ci, [self.num_rbfs, 1])
        w = np.reshape(self.rbfs_hi, [self.num_rbfs, 1])
        
        xi = w * (self.s_track - c) * (self.s_track - c)
        psi_track = np.exp(- xi) #+ sys.float_info.min

        self.psi_track = np.nan_to_num(psi_track)
        return self.psi_track

    def rbfs_step(self, s_step):
        
        psi_step = np.zeros(self.num_rbfs)

        for i in range(self.num_rbfs):
            
            xi = self.rbfs_hi[i] * (s_step - self.rbfs_ci[i]) * (s_step - self.rbfs_ci[i])
            psi_step[i] = np.exp(- xi) # + sys.float_info.min

        self.psi_step = psi_step
        
        return self.psi_step