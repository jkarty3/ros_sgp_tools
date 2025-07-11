import os
import yaml
import sys
import psutil
from time import gmtime, strftime

from launch_ros.actions import Node
from launch import LaunchDescription
from launch_ros.actions import PushRosNamespace
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import GroupAction, IncludeLaunchDescription
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource


def get_var(var, default):
    try:
        return os.environ[var]
    except:
        return default

def generate_launch_description():
    # Sanity check to avoid crashing the system because of low memory
    total_memory = psutil.virtual_memory().total + psutil.swap_memory().total
    if total_memory < 4e9:
        sys.exit("Low memory (less than 4GB), increase swap size!")

    # Get parameter values
    namespace = get_var('NAMESPACE', 'robot_0')
    data_type = get_var('DATA_TYPE' ,'GazeboPing1D')
    num_waypoints = int(get_var('NUM_WAYPOINTS', 20))
    sampling_rate = int(get_var('SAMPLING_RATE', 2))
    data_buffer_size = int(get_var('DATA_BUFFER_SIZE', 200))
    train_param_inducing = True if get_var('TRAIN_PARAM_INDUCING', 'False')=='True' else False
    num_param_inducing = int(get_var('NUM_PARAM_INDUCING', 40))
    adaptive_ipp = True if get_var('ADAPTIVE_IPP', 'True')=='True' else False
    data_folder = get_var('DATA_FOLDER', '')
    fcu_url = get_var('FCU_URL', 'udp://0.0.0.0:14550@')
    ping1d_port = get_var('PING1D_PORT', '/dev/ttyUSB0')
    kernel = get_var('KERNEL', 'RBF')
    distance_budget = float(get_var('DISTANCE_BUDGET', 100))

    num_robots = 1
    geofence_plan = PathJoinSubstitution([FindPackageShare('ros_sgp_tools'),
                                          'launch', 'data',
                                          'mission.plan'])
    
    print("\nParameters:")
    print("===========")
    print(f"DATA_TYPE: {data_type}")
    print(f"NUM_WAYPOINTS: {num_waypoints}")
    print(f"SAMPLING_RATE: {sampling_rate}")
    print(f"DIZTANCE_BUDGET: {distance_budget}")
    print(f"DATA_BUFFER_SIZE: {data_buffer_size}")
    print(f"TRAIN_PARAM_INDUCING: {train_param_inducing}")
    print(f"NUM_PARAM_INDUCING': {num_param_inducing}")
    print(f"ADAPTIVE_IPP: {adaptive_ipp}")
    print(f"KERNEL: {kernel}")
    print(f"FCU_URL: {fcu_url}")
    if data_type=='Ping1D':
        print(f"PING1D_PORT: {ping1d_port}")

    # Create data folder
    time_stamp = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    data_folder = os.path.join(data_folder, f'IPP-mission-{time_stamp}')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    
    # Dump parameters to YAML file
    params_dict = {
	    'namespace': namespace,
	    'data_type': data_type,
	    'num_waypoints': num_waypoints,
	    'sampling_rate': sampling_rate,
        'distance_budget': distance_budget,
	    'data_buffer_size': data_buffer_size,
	    'train_param_inducing': train_param_inducing,
	    'num_param_inducing': num_param_inducing,
	    'adaptive_ipp': adaptive_ipp,
	    'data_folder': data_folder,
	    'fcu_url': fcu_url,
	    'ping1d_port': ping1d_port,
	    'kernel': kernel,
	}
    ros_param_yaml = {
        namespace + '/IPP_params': {
            'ros__parameters': params_dict
        }
    }
    yaml_output_path = os.path.join(data_folder, f'launch_params.yaml')
    with open(yaml_output_path, 'w') as f:
        yaml.dump(ros_param_yaml, f)
    print(f"Parameters saved to {yaml_output_path}")
    print('')

    nodes = []

    # Offline IPP for initial path
    offline_kernel = 'None' if kernel == 'None' else 'RBF'
    offline_planner = Node(package='ros_sgp_tools',
                           executable='offline_ipp.py',
                           name='OfflineIPP',
                           parameters=[
                                {'num_waypoints': num_waypoints,
                                 'num_robots': num_robots,
                                 'sampling_rate': sampling_rate,
                                 'geofence_plan': geofence_plan,
                                 'kernel': offline_kernel,
                                 'distance_budget': distance_budget
                                }
                           ])
    nodes.append(offline_planner)

    # Online/Adaptive IPP
    online_planner = Node(package='ros_sgp_tools',
                          executable='online_ipp.py',
                          namespace=namespace,
                          name='OnlineIPP',
                          parameters=[
                              {'data_type': data_type,
                               'adaptive_ipp': adaptive_ipp,
                               'distance_budget': distance_budget,
                               'data_folder': data_folder,
                               'data_buffer_size': data_buffer_size,
                               'train_param_inducing': train_param_inducing,
                               'num_param_inducing': num_param_inducing,
                               'kernel': kernel
                              }
                          ])
    nodes.append(online_planner)

    # MAVROS controller
    path_follower = Node(package='ros_sgp_tools',
                         executable='path_follower.py',
                         namespace=namespace,
                         parameters=[{'xy_tolerance': 1.0,
                                      'geofence_plan': geofence_plan,
                                      'sampling_rate': sampling_rate}],
                         name='PathFollower')
    nodes.append(path_follower)

    # MAVROS
    mavros = GroupAction(
                    actions=[
                        PushRosNamespace(namespace),
                        IncludeLaunchDescription(
                            XMLLaunchDescriptionSource([
                                PathJoinSubstitution([
                                    FindPackageShare('mavros_control'),
                                    'launch',
                                    'mavros.launch'
                                ])
                            ]),
                            launch_arguments={
                                "fcu_url": fcu_url
                            }.items()
                        ),
                    ]
                )
    nodes.append(mavros)

    if data_type=='Ping1D':
        # Ping1D ROS package 
        sensor = Node(package='ping_sonar_ros',
                      executable='ping1d_node',
                      name='Ping1D',
                      namespace=namespace,
                      parameters=[
                        {'port': ping1d_port}
                      ])
        nodes.append(sensor)   
    elif data_type=='GazeboPing1D':
        # Gazebo ROS Bridge
        bridge = Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[f'ping1d@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan'],
            namespace=namespace
        )
        nodes.append(bridge)         

    return LaunchDescription(nodes)
