from launch_ros.actions import Node
from launch import LaunchDescription


def generate_launch_description():
    start_server = Node(
        package="dofbot_moveit", executable="dofbot_server", output="screen"
    )

    start_model = Node(
        package="robot_arm_color_stacking", executable="color_stacking", output="screen"
    )

    return LaunchDescription(
        [
            start_server,
            start_model,
        ]
    )
