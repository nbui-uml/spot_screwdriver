import argparse
import sys
import time
import math

import bosdyn.api.gripper_command_pb2
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry
from bosdyn.api import arm_command_pb2, geometry_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME,  HAND_FRAME_NAME, get_a_tform_b
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.image import ImageClient

class ArmClient:
    def __init__(self, config) -> None:
        bosdyn.client.util.setup_logging(config.verbose)
        self.sdk = bosdyn.client.create_standard_sdk("SpotArmClient")
        self.robot = self.sdk.create_robot(config.hostname)
        bosdyn.client.util.authenticate(self.robot)
        self.robot.time_sync.wait_for_sync()


    def power_off_safe(self):
        # Power the robot off. By specifying "cut_immediately=False", a safe power off command
        # is issued to the robot. This will attempt to sit the robot before powering off.
        self.robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not self.robot.is_powered_on(), "Robot power off failed."
        self.robot.logger.info("Robot safely powered off.")


    def arm_to_front(self):
        """
        Brings the arm to the front of the robot.
        """
        robot = self.robot

        assert robot.has_arm(), "Robot requires an arm to run this client."

        # Verify the robot is not estopped and that an external application has registered and holds
        # an estop endpoint.
        assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                        "such as the estop SDK example, to configure E-Stop."

        robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)

        # Now, we are ready to power on the robot. This call will block until the power
        # is on. Commands would fail if this did not happen. We can also check that the robot is
        # powered at any point.
        if not robot.is_powered_on():
            robot.logger.info("Powering on robot... This may take a several seconds.")
            robot.power_on(timeout_sec=20)
            assert robot.is_powered_on(), "Robot power on failed."
            robot.logger.info("Robot powered on.")

        # Tell the robot to stand up. The command service is used to issue commands to a robot.
        # The set of valid commands for a robot depends on hardware configuration. See
        # SpotCommandHelper for more detailed examples on command building. The robot
        # command service requires timesync between the robot and the client.
        robot.logger.info("Commanding robot to stand...")
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info("Robot standing.")

        # Make the arm pose RobotCommand
        # Build a position to move the arm to (in meters, relative to and expressed in the gravity aligned body frame).
        x = 0.55
        y = 0
        z = -0.10
        hand_ewrt_flat_body = geometry_pb2.Vec3(x=x, y=y, z=z)

        # Rotation as a quaternion
        q = bosdyn.geometry.EulerZXY(math.radians(180), 0.0, math.radians(15)).to_quaternion() #whatever angle spot's cameras are looking down at
        flat_body_Q_hand = geometry_pb2.Quaternion(w=q.w, x=q.x, y=q.y, z=q.z)

        flat_body_T_hand = geometry_pb2.SE3Pose(position=hand_ewrt_flat_body,
                                                rotation=flat_body_Q_hand)

        robot_state = robot_state_client.get_robot_state()
        odom_T_flat_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                        ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

        odom_T_hand = odom_T_flat_body * math_helpers.SE3Pose.from_obj(flat_body_T_hand)

        # duration in seconds
        seconds = 3

        arm_command = RobotCommandBuilder.arm_pose_command(
            odom_T_hand.x, odom_T_hand.y, odom_T_hand.z, odom_T_hand.rot.w, odom_T_hand.rot.x,
            odom_T_hand.rot.y, odom_T_hand.rot.z, ODOM_FRAME_NAME, seconds)

        # Send the request
        cmd_id = command_client.robot_command(arm_command)
        robot.logger.info('Moving arm to viewing position.')

        # Wait until the arm arrives at the goal.
        block_until_arm_arrives(command_client, cmd_id, seconds + 3.0)

        robot.logger.info('Done.')


    def arm_to_pose(self, x, y, z, Z, X, Y, relative_frame=GRAV_ALIGNED_BODY_FRAME_NAME):
        """
        Directs the arm to a specified pose.
        @param x, y, z: type float, position relative to relative_frame
        @param Z, X, Y: type float, orientation in Euler angles in degrees
        @param relative_frame: type string, name of the frame that the pose is based on. default=GRAV_ALIGNED_BODY_FRAME_NAME
        """
        robot = self.robot

        assert robot.has_arm(), "Robot requires an arm to run this client."

        # Verify the robot is not estopped and that an external application has registered and holds
        # an estop endpoint.
        assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                        "such as the estop SDK example, to configure E-Stop."

        robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)

        if not robot.is_powered_on():
            robot.logger.info("Powering on robot... This may take a several seconds.")
            robot.power_on(timeout_sec=20)
            assert robot.is_powered_on(), "Robot power on failed."
            robot.logger.info("Robot powered on.")

        # Tell the robot to stand up. The command service is used to issue commands to a robot.
        # The set of valid commands for a robot depends on hardware configuration. See
        # SpotCommandHelper for more detailed examples on command building. The robot
        # command service requires timesync between the robot and the client.
        robot.logger.info("Commanding robot to stand...")
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=10)

        #Need to get current pose
        robot_state = robot_state_client.get_robot_state()

        # Make the arm pose RobotCommand
        # Build a position to move the arm to (in meters, relative to and expressed in the gravity aligned body frame).
        hand_ewrt_flat_body = geometry_pb2.Vec3(x=x, y=y, z=z)

        # Rotation as a quaternion
        q = bosdyn.geometry.EulerZXY(math.radians(Z), math.radians(X), math.radians(Y)).to_quaternion()

        flat_body_Q_hand = geometry_pb2.Quaternion(w=q.w, x=q.x, y=q.y, z=q.z)

        flat_body_T_hand = geometry_pb2.SE3Pose(position=hand_ewrt_flat_body,
                                                rotation=flat_body_Q_hand)
        odom_T_flat_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                        ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

        odom_T_hand = odom_T_flat_body * math_helpers.SE3Pose.from_obj(flat_body_T_hand)

        # duration in seconds
        seconds = 3

        arm_command = RobotCommandBuilder.arm_pose_command(
            odom_T_hand.x, odom_T_hand.y, odom_T_hand.z, odom_T_hand.rot.w, odom_T_hand.rot.x,
            odom_T_hand.rot.y, odom_T_hand.rot.z, ODOM_FRAME_NAME, seconds)

        # Send the request
        cmd_id = command_client.robot_command(arm_command)
        robot.logger.info('Moving arm to viewing position.')

        # Wait until the arm arrives at the goal.
        block_until_arm_arrives(command_client, cmd_id, seconds + 3.0)

        robot.logger.info('Done.')

    def stow_arm(self):
        robot = self.robot

        assert robot.has_arm(), "Robot requires an arm to run this client."

        # Verify the robot is not estopped and that an external application has registered and holds
        # an estop endpoint.
        assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                        "such as the estop SDK example, to configure E-Stop."

        robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        
        if not robot.is_powered_on():
            robot.logger.info("Powering on robot... This may take a several seconds.")
            robot.power_on(timeout_sec=20)
            assert robot.is_powered_on(), "Robot power on failed."
            robot.logger.info("Robot powered on.")

        stow = RobotCommandBuilder.arm_stow_command()
        stow_command_id = command_client.robot_command(stow)
        robot.logger.info("Stow command issued.")
        block_until_arm_arrives(command_client, stow_command_id, 3.0)

def main(argv):
    """Command line interface."""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(argv)
    try: #get lease here and do stuff
        arm_client = ArmClient(options)
        lease_client = arm_client.robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
        with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            #arm to front
            arm_client.arm_to_front()
            time.sleep(3)

            #arm to right cam
            front_pose =  {
                "position" : (0.55, 0.0, -0.10),
                "orientation" : (0.0, 15.0, 180.0)
            }
            x, y, z = front_pose["position"]
            X, Y, Z = front_pose["orientation"]
            Z = 180 + 10
            arm_client.arm_to_pose(x, y, z, Z, X, Y)

            #stow arm and power off
            arm_client.stow_arm()
            arm_client.power_off_safe()
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.exception("Threw an exception")
        return False


if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)