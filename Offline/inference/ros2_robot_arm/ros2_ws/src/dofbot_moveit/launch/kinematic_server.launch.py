import os

import launch
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    server_param = launch.substitutions.LaunchConfiguration(
        'server_param',
        default = os.path.join(
            get_package_share_directory('dofbot_moveit'),
            'config',
            'server_param.yaml'
        )        
    )
    
    return LaunchDescription([
        Node(
            package = "dofbot_moveit",
            executable = "dofbot_server",
            output = "screen",
            parameters = [server_param]
        )
    ])