import os

from ament_index_python import get_package_share_directory
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, LaunchConfiguration


def generate_launch_description():
    res = [] 
    
    sensor_node = Node(
        package="mecharm_pi",
        executable="noisy_sensor",
        name="noisy_sensor",
        output="screen"
    )
    res.append(sensor_node)
    
    SOFC_GUI_node = Node(
        package="mecharm_pi",
        executable="SOFC_gui",
        name="SOFC_gui",
        output="screen"
    )
    res.append(SOFC_GUI_node)
    
    SOFC_node = Node(
        package="mecharm_pi",
        executable="SOFC",
        name="SOFC",
        output="screen"
    )
    res.append(SOFC_node)

    return LaunchDescription(res)
