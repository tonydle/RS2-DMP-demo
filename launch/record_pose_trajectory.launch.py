from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pose_topic = LaunchConfiguration("pose_topic")
    output_file = LaunchConfiguration("output_file")
    use_msg_stamp = LaunchConfiguration("use_msg_stamp")
    min_translation_delta = LaunchConfiguration("min_translation_delta")
    min_rotation_delta_rad = LaunchConfiguration("min_rotation_delta_rad")
    max_samples = LaunchConfiguration("max_samples")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "pose_topic",
                default_value="/vr_controller/pose",
                description="PoseStamped topic to record.",
            ),
            DeclareLaunchArgument(
                "output_file",
                default_value="/tmp/recorded_trajectory.txt",
                description="Path to the generated trajectory file.",
            ),
            DeclareLaunchArgument(
                "use_msg_stamp",
                default_value="true",
                description="Use PoseStamped header timestamps when available.",
            ),
            DeclareLaunchArgument(
                "min_translation_delta",
                default_value="0.0",
                description="Minimum translation change in meters before storing a sample.",
            ),
            DeclareLaunchArgument(
                "min_rotation_delta_rad",
                default_value="0.0",
                description="Minimum orientation change in radians before storing a sample.",
            ),
            DeclareLaunchArgument(
                "max_samples",
                default_value="0",
                description="Stop automatically after this many samples. 0 disables the limit.",
            ),
            Node(
                package="rs2_dmp_demo",
                executable="record_pose_trajectory",
                name="record_pose_trajectory",
                output="screen",
                parameters=[
                    {
                        "pose_topic": ParameterValue(pose_topic, value_type=str),
                        "output_file": ParameterValue(output_file, value_type=str),
                        "use_msg_stamp": ParameterValue(use_msg_stamp, value_type=bool),
                        "min_translation_delta": ParameterValue(
                            min_translation_delta, value_type=float
                        ),
                        "min_rotation_delta_rad": ParameterValue(
                            min_rotation_delta_rad, value_type=float
                        ),
                        "max_samples": ParameterValue(max_samples, value_type=int),
                    }
                ],
            ),
        ]
    )
