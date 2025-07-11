#! /usr/bin/env python3

from mavros_control.controller import Controller
from ros_sgp_tools.srv import Waypoints
from ros_sgp_tools.msg import ETA
import numpy as np
import rclpy
import os
from ament_index_python.packages import get_package_share_directory
from utils import get_mission_plan, calculate_bounded_path
from shapely.geometry import Polygon, LineString, Point


class IPPPathFollower(Controller):

    def __init__(self):
        super().__init__(navigation_type=0,
                         start_mission=False)

        # Setup current waypoint publisher that publishes at 10Hz
        self.eta_publisher = self.create_publisher(ETA, 'eta', 10)

        # Setup waypoint service
        self.waypoint_service = self.create_service(Waypoints, 
                                                    'waypoints',
                                                    self.waypoint_service_callback)
        
        # Get the mission boundary
        plan_fname = os.path.join(get_package_share_directory('ros_sgp_tools'), 
                                                        'launch', 
                                                        'sample.plan')
        self.declare_parameter('geofence_plan', plan_fname)
        plan_fname = self.get_parameter('geofence_plan').get_parameter_value().string_value
        self.get_logger().info(f'New file test Plan File: {plan_fname}')
        self.fence_vertices, *_ = get_mission_plan(plan_fname,
                                                                  get_waypoints=False)
        if not np.array_equal(self.fence_vertices[0], self.fence_vertices[-1]):
            fence_vertices = np.vstack([self.fence_vertices, self.fence_vertices[0]])
        self.fence_polygon = Polygon(fence_vertices)

        self.declare_parameter('sampling_rate', 2)
        self.sampling_rate = self.get_parameter('sampling_rate').get_parameter_value().integer_value
        self.get_logger().info(f'Sampling Rate: {self.sampling_rate}')

        # Initialize variables
        self.waypoints = None
        self.eta_msg = ETA()
        self.eta_msg.current_waypoint = -1

        self.get_logger().info("Initialized, waiting for waypoints")

        # Wait to get the waypoints from the online IPP planner
        while rclpy.ok() and self.waypoints is None:
            rclpy.spin_once(self, timeout_sec=1.0)

        # Setup timers
        self.eta_timer = self.create_timer(1, self.publish_eta)

        # Start visiting the waypoints
        self.mission()

    def waypoint_service_callback(self, request, response):
        waypoints_msg = request.waypoints.waypoints

        waypoints = []
        for i in range(len(waypoints_msg)):
            waypoints.append([waypoints_msg[i].x, 
                              waypoints_msg[i].y,
                              waypoints_msg[i].z])
        waypoints = np.array(waypoints)

        # Check if the vehicle has already passed some updated waypoints
        if self.waypoints is not None:
            idx = self.eta_msg.current_waypoint
            delta = self.waypoints[:idx+1]-waypoints[:idx+1]
            if np.sum(np.abs(delta)) > 0:
                self.get_logger().info('Waypoints rejected! Vehicle has already passed some updated waypoints')
                self.get_logger().info(f'{delta}')
                response.success = False
                return response
        
        self.waypoints = waypoints
        self.distances = self.haversine(self.waypoints[1:], 
                                        self.waypoints[:-1])
        self.get_logger().info('Waypoints received and accepted')
        response.success = True
        return response
    
    def publish_eta(self):
        idx = self.eta_msg.current_waypoint-1
        if idx < 0:
            return
        self.distances[idx] = self.waypoint_distance
        waypoints_eta = self.distances/self.heading_velocity
        self.eta_msg.eta = []
        for i in range(len(waypoints_eta)):
            eta = -1. if i < idx else np.sum(waypoints_eta[idx:i+1])
            self.eta_msg.eta.append(eta)
        self.eta_publisher.publish(self.eta_msg)

    def mission(self):
        """IPP mission"""
        mission_altitude = self.vehicle_position[2] # Ignored by Rover

        self.get_logger().info('Engaging GUIDED mode')
        if self.engage_mode('GUIDED'):
            self.get_logger().info('GUIDED mode Engaged')

        self.get_logger().info('Setting current positon as home')
        if self.set_home(self.vehicle_position[0], self.vehicle_position[1]):
            self.get_logger().info('Home position set')

        self.get_logger().info('Arming')
        if self.arm(True):
            self.get_logger().info('Armed')

        if self.use_altitude:
            if self.tol_command(mission_altitude+20.0):
                self.get_logger().info('Takeoff complete')

        for i in range(len(self.waypoints)):
            self.eta_msg.current_waypoint = i

            # Check if the line from previous waypoint to current waypoint is within the boundary
            if i > 0:
                point_a = (self.waypoints[i-1][0], self.waypoints[i-1][1])
                point_b = (self.waypoints[i][0], self.waypoints[i][1])
                line = LineString([point_a, point_b])

                if not self.fence_polygon.contains(line):
                    self.get_logger().info(f'Planned path to waypoint {i} leaves the boundary. Replanning.')
                    bounded_path = calculate_bounded_path(point_a, point_b,self.fence_polygon, self.sampling_rate)
                    self.get_logger().info(f'Bounded path: {bounded_path}')

                    for j, point in enumerate(bounded_path):
                        self.get_logger().info(f'Visiting waypoint {i-1}.{j+1}: {point}')
                        if self.go2waypoint([point[0],
                                    point[1],
                                    self.waypoints[i][2]+mission_altitude]):
                            self.get_logger().info(f'Reached waypoint {i-1}.{j+1}')


            self.get_logger().info(f'Visiting waypoint {i}: {self.waypoints[i]}')
            if self.go2waypoint([self.waypoints[i][0],
                            self.waypoints[i][1],
                            self.waypoints[i][2]+mission_altitude]):
                self.get_logger().info(f'Reached waypoint {i}')
                

        if self.use_altitude:
            if self.tol_command(mission_altitude):
                self.get_logger().info('Landing complete')

        self.get_logger().info('Disarming')
        if self.arm(False):
            self.get_logger().info('Disarmed')

        self.get_logger().info('Mission complete')


def main(args=None):
    rclpy.init(args=args)
    path_follower = IPPPathFollower()
    rclpy.spin_once(path_follower)

if __name__ == '__main__':
    main()