#!/usr/bin/env python3
"""
ROS 2 node that uses CDMPs to generate a new Cartesian trajectory based on a demo trajectory,
starting from the current robot pose obtained from tf.
The node computes a linear demo trajectory relative to the current pose (using a user-defined
direction and distance), learns a CDMP from it, and then rolls out a reproduction trajectory with
optional collision avoidance.
Velocity commands are published on /servo_node/delta_twist_cmds.

User-configurable parameters at the top:
  - use_collision: Whether a collision object is in place (activates obstacle avoidance)
  - movement_direction: The axis/direction of movement (e.g. [1, 0, 0] for x-axis)
  - movement_distance: The distance (in meters) of the linear movement
  - ee_link: The end-effector link name (used to look up its pose from tf)
  - base_link: The base link name (used as the reference frame for tf and as the twist header)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from visualization_msgs.msg import Marker
import numpy as np
import math
import time

import tf2_ros
from tf2_ros import TransformException

# --- Import the CDMPs class (make sure its module is in your PYTHONPATH) ---
from cdmp.src.cdmps import CartesianDMPs


class CDMPServoNode(Node):
    def __init__(self):
        super().__init__('cdmp_servo_node')

        # =================== User Parameters =================== #
        # Collision avoidance activation
        self.use_collision = True  

        # Movement parameters (relative movement direction and distance)
        self.movement_direction = np.array([1.0, 0.0, 0.0])  # e.g., along -y-axis
        self.movement_distance = 1.0  # in meters
        self.movement_time = 10.0  # in seconds (for demo trajectory generation)
        self.num_demo_points = 100  # number of demo points for interpolation

        # TF parameters: set the end-effector link and base link names
        self.ee_link = "panda_link8"      # adjust as needed
        self.base_link = "panda_link0"    # adjust as needed
        # ========================================================= #

        # --- CDMP Parameters ---
        self.alpha_s_val = 7.0
        self.k_gain_val = 1000.0
        self.rbfs_pSec_val = 4.0
        self.tau_scaler_val = 1.0

        # Obstacle avoidance parameters
        self.beta_val = 3.0
        self.lambda_f_val = 3.0
        self.eta_val = 3.0
        self.obstacle_pos = np.array([0.45, 0.1, 0.6])

        # Publisher for TwistStamped messages
        self.vel_pub = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)

        # Publisher for the collision object marker
        self.marker_pub = self.create_publisher(Marker, 'visualization_marker', 10)
        # Create a timer to publish the marker periodically
        self.marker_timer = self.create_timer(1.0, self.publish_marker_callback)

        # Initialize tf2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info("CDMP Servo Node started. Waiting for TF transform...")

        # CDMP instance (to be initialized after obtaining current pose)
        self.cdmp = None
        self.dt = None
        self.step_index = 0
        self.servo_timer = None

        # Start a timer to try to get the current pose from TF
        self.tf_timer = self.create_timer(0.1, self.tf_timer_callback)

    def publish_marker_callback(self):
        """
        Publishes a visualization marker representing the collision object.
        """
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.base_link
        marker.ns = "collision_object"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = self.obstacle_pos[0]
        marker.pose.position.y = self.obstacle_pos[1]
        marker.pose.position.z = self.obstacle_pos[2]
        marker.pose.orientation.w = 1.0  # no rotation
        # Set the sphere size (adjust as needed)
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        # Set color (red, semi-transparent)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        marker.lifetime.sec = 0  # 0 means marker never auto-deletes

        self.marker_pub.publish(marker)

    def tf_timer_callback(self):
        try:
            now = rclpy.time.Time()
            # Lookup the transform from base_link to ee_link (i.e. current EE pose in base frame)
            trans = self.tf_buffer.lookup_transform(self.base_link, self.ee_link, now)
            # Extract position and orientation from the transform message
            self.start_pos = np.array([trans.transform.translation.x,
                                       trans.transform.translation.y,
                                       trans.transform.translation.z])
            self.start_quat = np.array([trans.transform.rotation.w,
                                        trans.transform.rotation.x,
                                        trans.transform.rotation.y,
                                        trans.transform.rotation.z])
            self.get_logger().info(f"Got current pose from TF: {self.start_pos}")

            # Compute the goal position by adding a relative displacement
            norm_dir = self.movement_direction / np.linalg.norm(self.movement_direction)
            self.goal_pos = self.start_pos + norm_dir * self.movement_distance
            self.goal_quat = self.start_quat.copy()

            # Print the start and goal positions
            self.get_logger().info(f"Start position: {self.start_pos}")
            self.get_logger().info(f"Goal position: {self.goal_pos}")

            # --- Generate a Demo Trajectory (Linear Interpolation) ---
            self.dem_time = np.linspace(0, self.movement_time, self.num_demo_points)
            self.dt = self.dem_time[1] - self.dem_time[0]
            # Interpolate positions between start and goal
            self.dem_pos = np.linspace(self.start_pos, self.goal_pos, self.num_demo_points)
            # Use constant orientation (or interpolate if needed)
            self.dem_quat = np.tile(self.start_quat, (self.num_demo_points, 1))

            # --- Initialize and Learn CDMP from the Demo Trajectory ---
            self.cdmp = CartesianDMPs()
            self.cdmp.load_demo(filename="demo",
                                dem_time=self.dem_time,
                                dem_pos=self.dem_pos,
                                dem_quat=self.dem_quat)
            self.cdmp.learn_cdmp(alpha_s=self.alpha_s_val,
                                 k_gain=self.k_gain_val,
                                 rbfs_pSec=self.rbfs_pSec_val)
            # For reproduction, we use no offset (relative pose = zeros)
            rep_relpose_start = np.hstack((np.zeros(3), self.start_quat))
            rep_relpose_goal = np.hstack((np.zeros(3), self.goal_quat))
            self.cdmp.init_reproduction(tau_scaler=self.tau_scaler_val,
                                        rep_relpose_start=rep_relpose_start,
                                        rep_relpose_goal=rep_relpose_goal)

            # Start the servoing timer callback using the demo dt
            self.servo_timer = self.create_timer(self.dt, self.servo_callback)
            self.get_logger().info("CDMP initialized. Starting servoing...")
            # Cancel the TF timer since initialization is complete
            self.tf_timer.cancel()
        except TransformException as ex:
            self.get_logger().warn("TF lookup failed: " + str(ex))
        except Exception as e:
            self.get_logger().error("Error in TF lookup: " + str(e))

    def servo_callback(self):
        if self.step_index < len(self.cdmp.rep_cs.s_track):
            s_step = self.cdmp.rep_cs.s_track[self.step_index]

            # --- Obstacle Avoidance (if enabled) ---
            curr_pos = self.cdmp.rep_pos[-1]
            curr_linVel = self.cdmp.rep_linVel[-1]
            rep_force = np.zeros(3)
            if self.use_collision:
                d = np.linalg.norm(curr_pos - self.obstacle_pos)
                if d > 0:
                    nabla_d = (curr_pos - self.obstacle_pos) / d
                else:
                    nabla_d = np.zeros(3)
                curr_linVel_norm = np.linalg.norm(curr_linVel)
                if curr_linVel_norm > 0 and d > 0:
                    cos_theta = np.dot(curr_linVel, nabla_d) / (curr_linVel_norm * np.linalg.norm(nabla_d))
                    theta = math.acos(np.clip(cos_theta, -1.0, 1.0))
                    if theta > (math.pi / 2) and theta <= math.pi:
                        rep_force = ((-cos_theta) ** self.beta_val * self.lambda_f_val *
                                     self.eta_val * curr_linVel_norm * nabla_d) / (d ** (self.eta_val + 1))

            # --- CDMP Rollout Step ---
            next_pos, next_linVel, next_quat = self.cdmp.rollout_step(curr_s_step=s_step,
                                                                      ext_force=rep_force)
            lin_vel = next_linVel[0]

            # Create and populate a TwistStamped message for the velocity command
            twist = TwistStamped()
            twist.header.stamp = self.get_clock().now().to_msg()
            twist.header.frame_id = self.base_link
            twist.twist.linear.x = lin_vel[0]
            twist.twist.linear.y = lin_vel[1]
            twist.twist.linear.z = lin_vel[2]
            twist.twist.angular.x = 0.0
            twist.twist.angular.y = 0.0
            twist.twist.angular.z = 0.0

            self.vel_pub.publish(twist)
            self.step_index += 1
        else:
            self.get_logger().info("CDMP trajectory complete. Stopping servoing.")
            self.servo_timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    node = CDMPServoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down CDMP servo node.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
