#!/usr/bin/env python3

from pathlib import Path
from typing import List, Optional, Tuple

import math

from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.node import Node


CSV_HEADER = "TimeStep,PosX,PosY,PosZ,QuatW,QuatX,QuatY,QuatZ"


class PoseTrajectoryRecorder(Node):
    def __init__(self) -> None:
        super().__init__("pose_trajectory_recorder")

        self.declare_parameter("pose_topic", "/vr_controller/pose")
        self.declare_parameter("output_file", "/tmp/recorded_trajectory.txt")
        self.declare_parameter("use_msg_stamp", True)
        self.declare_parameter("min_translation_delta", 0.0)
        self.declare_parameter("min_rotation_delta_rad", 0.0)
        self.declare_parameter("max_samples", 0)

        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.output_file = str(self.get_parameter("output_file").value)
        self.use_msg_stamp = bool(self.get_parameter("use_msg_stamp").value)
        self.min_translation_delta = float(
            self.get_parameter("min_translation_delta").value
        )
        self.min_rotation_delta = float(
            self.get_parameter("min_rotation_delta_rad").value
        )
        self.max_samples = int(self.get_parameter("max_samples").value)

        self.first_stamp_sec: Optional[float] = None
        self.samples: List[Tuple[float, float, float, float, float, float, float, float]] = []
        self.last_pose: Optional[Tuple[float, float, float, float, float, float, float]] = None
        self.saved = False

        self.subscription = self.create_subscription(
            PoseStamped,
            self.pose_topic,
            self.pose_callback,
            100,
        )

        self.get_logger().info(f"Recording PoseStamped messages from '{self.pose_topic}'")
        self.get_logger().info(f"Output file: {self.output_file}")

    def pose_callback(self, msg: PoseStamped) -> None:
        stamp_sec = self.get_stamp_seconds(msg)
        if self.first_stamp_sec is None:
            self.first_stamp_sec = stamp_sec

        rel_time = stamp_sec - self.first_stamp_sec
        pose = msg.pose
        quat = self.normalize_quaternion(
            pose.orientation.w,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
        )
        current_pose = (
            pose.position.x,
            pose.position.y,
            pose.position.z,
            quat[0],
            quat[1],
            quat[2],
            quat[3],
        )

        if self.should_skip_sample(current_pose):
            return

        self.samples.append((rel_time, *current_pose))
        self.last_pose = current_pose

        if len(self.samples) == 1:
            self.get_logger().info("Received first pose sample; recording started.")

        if self.max_samples > 0 and len(self.samples) >= self.max_samples:
            self.get_logger().info("Reached max_samples limit, stopping recorder.")
            self.write_samples_to_file()
            self.destroy_node()
            rclpy.shutdown()

    def get_stamp_seconds(self, msg: PoseStamped) -> float:
        if self.use_msg_stamp:
            stamp = msg.header.stamp
            if stamp.sec != 0 or stamp.nanosec != 0:
                return float(stamp.sec) + float(stamp.nanosec) * 1e-9

        return self.get_clock().now().nanoseconds * 1e-9

    def should_skip_sample(
        self, current_pose: Tuple[float, float, float, float, float, float, float]
    ) -> bool:
        if self.last_pose is None:
            return False

        dx = current_pose[0] - self.last_pose[0]
        dy = current_pose[1] - self.last_pose[1]
        dz = current_pose[2] - self.last_pose[2]
        translation_delta = math.sqrt(dx * dx + dy * dy + dz * dz)

        rotation_delta = self.quaternion_angle(self.last_pose[3:], current_pose[3:])

        return (
            translation_delta < self.min_translation_delta
            and rotation_delta < self.min_rotation_delta
        )

    def quaternion_angle(
        self,
        quat_a: Tuple[float, float, float, float],
        quat_b: Tuple[float, float, float, float],
    ) -> float:
        dot = abs(
            quat_a[0] * quat_b[0]
            + quat_a[1] * quat_b[1]
            + quat_a[2] * quat_b[2]
            + quat_a[3] * quat_b[3]
        )
        dot = min(1.0, max(-1.0, dot))
        return 2.0 * math.acos(dot)

    def normalize_quaternion(
        self, w: float, x: float, y: float, z: float
    ) -> Tuple[float, float, float, float]:
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        if norm == 0.0:
            self.get_logger().warn("Received zero-length quaternion; defaulting to identity.")
            return (1.0, 0.0, 0.0, 0.0)
        return (w / norm, x / norm, y / norm, z / norm)

    def write_samples_to_file(self) -> None:
        if self.saved:
            return

        output_path = Path(self.output_file).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file_handle:
            file_handle.write(CSV_HEADER + "\n")
            for sample in self.samples:
                file_handle.write(",".join(f"{value:.6f}" for value in sample) + "\n")

        self.saved = True
        self.get_logger().info(
            f"Saved {len(self.samples)} samples to '{output_path}'"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PoseTrajectoryRecorder()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopping recorder and writing trajectory file.")
    finally:
        node.write_samples_to_file()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
