#!/usr/bin/env python3
"""
ROS 2 demo node that:
  1. Uses tf2 to get the current end-effector pose (from "ee_link" relative to "base_link").
  2. Generates a linear demo trajectory (by adding a relative displacement along a user-defined direction).
  3. Learns a CDMP from the demo trajectory.
  4. Rolls out the reproduction trajectory (optionally with collision avoidance).
  5. Publishes RViz markers for:
       - The original (demo) trajectory.
       - The reproduced trajectory.
       - The collision object.
  6. Finally, after visualization, publishes TwistStamped velocity commands to have the robot follow the trajectory.
  
User-configurable parameters:
  - use_collision: Activate collision-avoidance (and so display the collision object).
  - movement_direction: A 3D vector in the base frame.
  - movement_distance: Linear displacement (m) to add to the current position.
  - movement_time: The total demo duration in seconds.
  - ee_link: The end-effector link name (TF lookup).
  - base_link: The base link name (TF lookup and marker header).
  
This script is meant to run once as a demo.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, Point
from visualization_msgs.msg import Marker
import numpy as np
import math
import time

import tf2_ros
from tf2_ros import TransformException

# --- Import the CDMPs class ---
from cdmp.src.cdmps import CartesianDMPs

class CDMPDemoNode(Node):
    def __init__(self):
        super().__init__('cdmp_demo_node')

        # =================== User Parameters =================== #
        self.use_collision = True  # if True, collision avoidance is enabled

        # Relative movement parameters (in base_link coordinates)
        self.movement_direction = np.array([1.0, 0.0, 0.0])  # e.g., along +x-axis
        self.movement_distance = 0.4  # in meters
        self.movement_time = 1.0     # demo trajectory duration (s)
        self.num_demo_points = 300    # number of interpolation points

        # TF parameters: end-effector and base link names
        self.ee_link = "panda_link8"    # adjust as needed
        self.base_link = "panda_link0"  # adjust as needed

        # --- CDMP Parameters ---
        self.alpha_s_val = 7.0
        self.k_gain_val = 100.0
        self.rbfs_pSec_val = 4.0
        self.tau_scaler_val = 1.0

        # Obstacle avoidance parameters and object position (in base_link frame)
        self.beta_val = 1.5
        self.lambda_f_val = 1.5
        self.eta_val = 1.5
        self.obstacle_pos = np.array([0.45, 0.05, 0.6])

        # Publishers for visualization markers and velocity commands
        self.marker_pub = self.create_publisher(Marker, 'visualization_marker', 10)
        self.vel_pub = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)

        # Initialize tf2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # CDMP instance (to be created later)
        self.cdmp = None

    def get_current_tf(self, timeout=5.0):
        """Block until a transform is available or timeout is reached."""
        self.get_logger().info("Waiting for TF transform from {} to {}..."
                               .format(self.base_link, self.ee_link))
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            try:
                now = rclpy.time.Time()
                transform = self.tf_buffer.lookup_transform(
                    self.base_link,
                    self.ee_link,
                    now,
                    rclpy.duration.Duration(seconds=0.5))
                return transform
            except TransformException as ex:
                self.get_logger().warn("TF lookup failed: " + str(ex))
                time.sleep(0.1)
        self.get_logger().error("Timeout waiting for TF transform.")
        raise RuntimeError("TF transform not available.")

    def run_cdmp_demo(self):
        # 1. Get the current end-effector pose from TF.
        transform = self.get_current_tf()
        self.start_pos = np.array([transform.transform.translation.x,
                                   transform.transform.translation.y,
                                   transform.transform.translation.z])
        self.start_quat = np.array([transform.transform.rotation.w,
                                    transform.transform.rotation.x,
                                    transform.transform.rotation.y,
                                    transform.transform.rotation.z])
        self.get_logger().info(f"Start pose: {self.start_pos}")

        # 2. Define the goal pose by adding the relative displacement.
        norm_dir = self.movement_direction / np.linalg.norm(self.movement_direction)
        self.goal_pos = self.start_pos + norm_dir * self.movement_distance
        self.goal_quat = self.start_quat.copy()
        self.get_logger().info(f"Goal pose: {self.goal_pos}")

        # 3. Generate the demo trajectory (linear interpolation from start to goal)
        self.dem_time = np.linspace(0, self.movement_time, self.num_demo_points)
        dt = self.dem_time[1] - self.dem_time[0]
        dem_pos = np.linspace(self.start_pos, self.goal_pos, self.num_demo_points)
        dem_quat = np.tile(self.start_quat, (self.num_demo_points, 1))

        # Publish marker for the demo trajectory
        self.publish_trajectory_marker(dem_pos, marker_id=1, ns="demo_path",
                                         color=(0.0, 0.0, 1.0, 1.0),  # blue
                                         marker_text="Demo Trajectory")

        # 4. Initialize and learn CDMP from the demo trajectory.
        cdmp = CartesianDMPs()
        cdmp.load_demo(filename="demo",
                       dem_time=self.dem_time,
                       dem_pos=dem_pos,
                       dem_quat=dem_quat)
        cdmp.learn_cdmp(alpha_s=self.alpha_s_val,
                        k_gain=self.k_gain_val,
                        rbfs_pSec=self.rbfs_pSec_val)
        rep_relpose_start = np.hstack((np.zeros(3), self.start_quat))
        rep_relpose_goal = np.hstack((np.zeros(3), self.goal_quat))
        cdmp.init_reproduction(tau_scaler=self.tau_scaler_val,
                               rep_relpose_start=rep_relpose_start,
                               rep_relpose_goal=rep_relpose_goal)

        # 5. Roll out the reproduction trajectory.
        rep_positions = []
        num_steps = len(cdmp.rep_cs.s_track)
        for i in range(num_steps):
            s_step = cdmp.rep_cs.s_track[i]
            # Compute repulsive force if collision avoidance is enabled
            rep_force = np.zeros(3)
            if self.use_collision:
                curr_pos = cdmp.rep_pos[-1]
                curr_linVel = cdmp.rep_linVel[-1]
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
            next_pos, next_linVel, next_quat = cdmp.rollout_step(curr_s_step=s_step,
                                                                  ext_force=rep_force)
            rep_positions.append(next_pos.flatten())
        rep_positions = np.array(rep_positions)

        # Publish marker for the reproduction trajectory
        self.publish_trajectory_marker(rep_positions, marker_id=2, ns="reproduced_path",
                                         color=(0.0, 1.0, 0.0, 1.0),  # green
                                         marker_text="Reproduced Trajectory")
        
        # Publish the collision object marker.
        self.publish_collision_marker()

        self.get_logger().info("Markers published.")
        input("Press Enter to start following the trajectory...")

        # 6. Make the robot follow the reproduced trajectory.
        # Compute approximate velocity commands as finite differences between positions.
        self.get_logger().info("Following the reproduced trajectory...")
        for i in range(len(rep_positions) - 1):
            # Compute velocity vector (position difference divided by dt)
            velocity = (rep_positions[i+1] - rep_positions[i]) / dt
            twist = TwistStamped()
            twist.header.stamp = self.get_clock().now().to_msg()
            twist.header.frame_id = self.base_link
            twist.twist.linear.x = velocity[0]
            twist.twist.linear.y = velocity[1]
            twist.twist.linear.z = velocity[2]
            twist.twist.angular.x = 0.0
            twist.twist.angular.y = 0.0
            twist.twist.angular.z = 0.0
            self.vel_pub.publish(twist)
            time.sleep(dt)
        # At the end, publish a zero command to stop the robot.
        stop_twist = TwistStamped()
        stop_twist.header.stamp = self.get_clock().now().to_msg()
        stop_twist.header.frame_id = self.base_link
        self.vel_pub.publish(stop_twist)
        self.get_logger().info("Trajectory following complete.")

    def publish_trajectory_marker(self, positions, marker_id, ns, color, marker_text):
        """
        Publishes a LINE_STRIP marker that visualizes the given set of positions.
        :param positions: Numpy array of shape (N, 3)
        :param marker_id: A unique ID for the marker.
        :param ns: The marker namespace.
        :param color: Tuple (r, g, b, a) for marker color.
        :param marker_text: Marker text (if used by TEXT_VIEW_FACING markers)
        """
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.base_link
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        for pos in positions:
            pt = Point()
            pt.x, pt.y, pt.z = pos
            marker.points.append(pt)
        marker.scale.x = 0.01  # line width
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        marker.lifetime.sec = 0
        marker.text = marker_text

        self.marker_pub.publish(marker)
        self.get_logger().info(f"Published {ns} marker with {len(positions)} points.")

    def publish_collision_marker(self):
        """Publishes a spherical marker representing the collision object."""
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
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        marker.lifetime.sec = 0

        self.marker_pub.publish(marker)
        self.get_logger().info("Published collision object marker.")


def main(args=None):
    rclpy.init(args=args)
    node = CDMPDemoNode()
    try:
        # Run the demo: generate paths, publish markers, and follow trajectory.
        node.run_cdmp_demo()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down CDMP demo node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
