import argparse
import sys
import time
import math

import bosdyn.api.gripper_command_pb2
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry
from bosdyn.api import arm_command_pb2, geometry_pb2, synchronized_command_pb2, robot_command_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME,  HAND_FRAME_NAME, get_a_tform_b
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient


def make_robot_command(arm_joint_traj: arm_command_pb2.ArmJointTrajectory) -> synchronized_command_pb2.SynchronizedCommand:
    """ Helper function to create a RobotCommand from an ArmJointTrajectory.
        The returned command will be a SynchronizedCommand with an ArmJointMoveCommand
        filled out to follow the passed in trajectory. """

    joint_move_command = arm_command_pb2.ArmJointMoveCommand.Request(trajectory=arm_joint_traj)
    arm_command = arm_command_pb2.ArmCommand.Request(arm_joint_move_command=joint_move_command)
    sync_arm = synchronized_command_pb2.SynchronizedCommand.Request(arm_command=arm_command)
    arm_sync_robot_cmd = robot_command_pb2.RobotCommand(synchronized_command=sync_arm)
    return RobotCommandBuilder.build_synchro_command(arm_sync_robot_cmd)


class ArmClient:
    def __init__(self, config: argparse.Namespace, robot: bosdyn.client.Robot) -> None:
        self.sdk = bosdyn.client.create_standard_sdk("SpotArmClient")
        self.robot = robot

        self.joint_states = {
            "front": (0.070,-0.328,1.611,-0.049,1.825,0.007)
        }


    def power_off_safe(self) -> None:
        """
        Commands the robot to perform a safe power off.
        """
        self.robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not self.robot.is_powered_on(), "Robot power off failed."
        self.robot.logger.info("Robot safely powered off.")

    def joint_move(self, sh0, sh1, el0, el1, wr0, wr1) -> None:
        """
        Directs the arm to a specified joint position
        @param sh0, sh1, el0, el1, wr0, wr1: type float, joint position
        """
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

        # Tell the robot to stand up.
        robot.logger.info("Commanding robot to stand...")
        blocking_stand(command_client, timeout_sec=10)

        # Make the arm pose RobotCommand
        traj_point = RobotCommandBuilder.create_arm_joint_trajectory_point(sh0,sh1,el0,el1,wr0,wr1)
        arm_joint_traj = arm_command_pb2.ArmJointTrajectory(points=[traj_point])
        command = make_robot_command(arm_joint_traj)

        #send request
        cmd_id = command_client.robot_command(command)
        robot.logger.info(f"Moving arm to position [{sh0},{sh1},{el0},{el1},{wr0},{wr1}].")

        block_until_arm_arrives(command_client, cmd_id, timeout_sec=10)
        print("Arm finished moving.")

    
    def arm_move(self, x, y, z, qw, qx, qy, qz,
                rframe: str, tf: geometry_pb2.FrameTreeSnapshot = None) -> None:
        """
        Directs the arm to move to a point in space
        @param x, y, z: type float, position in space relative to frame
        @param qw, qx, qy, qz: type float, rotation in space relative to frame
        @frame rframe: type str, name of the reference frame
        """
        robot = self.robot

        #check estop here
        assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                        "such as the estop SDK example, to configure E-Stop."
        assert robot.has_arm(), "Robot requires an arm to run this client."

        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)

        #power on
        if not robot.is_powered_on():
            robot.logger.info("Powering on robot... This may take a several seconds.")
            robot.power_on(timeout_sec=20)
            assert robot.is_powered_on(), "Robot power on failed."
            robot.logger.info("Robot powered on.")
        
        #stand
        robot.logger.info("Commanding robot to stand...")
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info("Robot standing.")

        hand_ewrt_rframe = geometry_pb2.Vec3(x=x, y=y, z=z)
        rframe_Q_hand = geometry_pb2.Quaternion(qw, qx, qy, qz)
        rframe_T_hand = geometry_pb2.SE3Pose(position=hand_ewrt_rframe, rotation=rframe_Q_hand)

        if tf is None:
            robot_state = robot_state_client.get_robot_state()
            tf = robot_state.kinematic_state.transforms_snapshot

        body_T_cam = get_a_tform_b(tf, GRAV_ALIGNED_BODY_FRAME_NAME, rframe)
        body_T_hand = body_T_cam * math_helpers.SE3Pose.from_obj(rframe_T_hand)

        arm_command = RobotCommandBuilder.arm_pose_command(
            body_T_hand.x, body_T_hand.y, body_T_hand.z,
            body_T_hand.rot.x, body_T_hand.rot.y, body_T_hand.rot.z, body_T_hand.rot.w, GRAV_ALIGNED_BODY_FRAME_NAME
        )

        cmd_id = command_client.robot_command(arm_command)
        robot.logger.info("Moving arm...")

        block_until_arm_arrives(command_client, cmd_id, 10)
        robot.logger.info("Finished moving.")


    def stow_arm(self) -> None:
        """
        Stows the arm
        """
        robot = self.robot

        assert robot.has_arm(), "Robot requires an arm to run this client."

        # Verify the robot is not estopped and that an external application has registered and holds
        # an estop endpoint.
        assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                        "such as the estop SDK example, to configure E-Stop."

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
        robot.logger.info("Arm stowed.")


#------Testing----------

def main(argv):
    """Command line interface."""
    from spot_docking_client import DockingClient

    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(argv)
    try: #get lease here and do stuff
        arm_client = ArmClient(options)
        docking_client = DockingClient(options)
        lease_client = arm_client.robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
        with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            #undock
            docking_client.undock()

            #arm to front
            sh0,sh1,el0,el1,wr0,wr1 = arm_client.joint_states["front"]
            arm_client.arm_to_pose(sh0,sh1,el0,el1,wr0,wr1)
            
            #wait
            input("Press any key to continue")

            #stow arm
            arm_client.stow_arm()

            #dock
            docking_client.dock(520)

            arm_client.power_off_safe()
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.exception("Threw an exception")
        return False


if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)