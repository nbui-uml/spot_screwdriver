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
from bosdyn.client import math_helpers, frame_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME,  HAND_FRAME_NAME, BODY_FRAME_NAME, get_a_tform_b
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.image import ImageClient , build_image_request


def make_robot_command(arm_joint_traj: arm_command_pb2.ArmJointTrajectory) -> synchronized_command_pb2.SynchronizedCommand:
    """ 
    Helper function to create a RobotCommand from an ArmJointTrajectory.
        The returned command will be a SynchronizedCommand with an ArmJointMoveCommand
        filled out to follow the passed in trajectory. 

    Parameters
    -----
    arm_joint_traj: ArmJointTrajectory

    Returns
    -----
    SynchronizedCommand    
    """

    joint_move_command = arm_command_pb2.ArmJointMoveCommand.Request(trajectory=arm_joint_traj)
    arm_command = arm_command_pb2.ArmCommand.Request(arm_joint_move_command=joint_move_command)
    sync_arm = synchronized_command_pb2.SynchronizedCommand.Request(arm_command=arm_command)
    arm_sync_robot_cmd = robot_command_pb2.RobotCommand(synchronized_command=sync_arm)
    return RobotCommandBuilder.build_synchro_command(arm_sync_robot_cmd)


class ArmClient:
    def __init__(self, config: argparse.Namespace, robot: bosdyn.client.Robot) -> None:
        self.config = config
        self.robot = robot

        self.joint_states = {
            "front": (0.0,-0.328,1.611,0.0,1.825,0.0),
            "frontright": (0.448,-0.328,1.611,-0.15,1.825,0.0),
            "frontleft": (-0.448,-0.328,1.611,0.15,1.825,0.0),
            "ready": (0.0, -1.57, 1.57, 0.0, 0.0, 0.0)
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
        Directs the arm to a specified joint position.
        
        Parameters
        -----
        sh0, sh1, el0, el1, wr0, wr1: float
            Joint position.
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
        Directs the arm to move to a point in space.

        Parameters
        -----
        x, y, z: float
            Position in space relative to frame.
        qw, qx, qy, qz: float
            Rotation in space relative to frame.
        rframe: str
            Name of the reference frame.
        tf: FrameTreeSnapshot
            Transform tree.
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
        rframe_Q_hand = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)
        rframe_T_hand = geometry_pb2.SE3Pose(position=hand_ewrt_rframe, rotation=rframe_Q_hand)

        if tf is None:
            robot_state = robot_state_client.get_robot_state()
            tf = robot_state.kinematic_state.transforms_snapshot

        body_T_cam = get_a_tform_b(tf, BODY_FRAME_NAME, rframe)
        
        body_T_hand = body_T_cam * math_helpers.SE3Pose.from_obj(rframe_T_hand)

        arm_command = RobotCommandBuilder.arm_pose_command(
            body_T_hand.x, body_T_hand.y, body_T_hand.z,
            body_T_hand.rot.w, body_T_hand.rot.x, body_T_hand.rot.y, body_T_hand.rot.z, BODY_FRAME_NAME
        )

        cmd_id = command_client.robot_command(arm_command)
        robot.logger.info("Moving arm...")

        block_until_arm_arrives(command_client, cmd_id, 10)
        robot.logger.info("Finished moving.")


    def stow_arm(self) -> None:
        """
        Stows the arm.
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

    
    def carry_position(self) -> None:
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

        carry = RobotCommandBuilder.arm_carry_command()
        carry_command_id = command_client.robot_command(carry)
        robot.logger.info("Carry command issued.")
        block_until_arm_arrives(command_client, carry_command_id, 3.0)
        robot.logger.info("Arm in carry position.")


#------Testing----------

def main(argv):
    """Command line interface."""
    from spot_docking_client import DockingClient
    from bosdyn.client.robot_state import RobotStateClient

    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(argv)

    bosdyn.client.util.setup_logging(options.verbose)
    sdk = bosdyn.client.create_standard_sdk("SpotArmClient")
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    
    assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                   "such as the estop SDK example, to configure E-Stop."
    try: #get lease here and do stuff
        lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
        robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        image_client = robot.ensure_client(ImageClient.default_service_name)

        arm_client = ArmClient(options, robot)
        docking_client = DockingClient(options, robot)
        with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            #undock
            #docking_client.undock()

            if not robot.is_powered_on():
                robot.logger.info("Powering on robot... This may take a several seconds.")
                robot.power_on(timeout_sec=20)
                assert robot.is_powered_on(), "Robot power on failed."
                robot.logger.info("Robot powered on.")

            robot.logger.info("Commanding robot to stand...")
            command_client = robot.ensure_client(RobotCommandClient.default_service_name)
            blocking_stand(command_client, timeout_sec=10)
            robot.logger.info("Robot standing.")

            '''
            #to front of camera
            robot.logger.info("Attempting to orient gripper in front of frontright fisheye camera.")
            image_responses = image_client.get_image([build_image_request("frontright_fisheye_image")])
            image = image_responses[0]
            print(bosdyn.client.frame_helpers.get_frame_names(image.shot.transforms_snapshot))
            arm_client.arm_move(0,0,0.5,0.707,0,0.707,0,"frontright_fisheye",image.shot.transforms_snapshot)

            input("Press any key to continue...")

            #stow arm
            arm_client.stow_arm()
            '''

            arm_client.carry_position()
            open_command = RobotCommandBuilder.claw_gripper_open_command()
            cmd_id = command_client.robot_command(open_command)
            robot.logger.info("Opening claw...")

            input("Press any key to continue...")

            close_command = RobotCommandBuilder.claw_gripper_close_command()
            cmd_id = command_client.robot_command(close_command)
            robot.logger.info("Grasping.")

            input("Press any key to continue...")


            #arm to front
            robot.logger.info("Moving arm to front")
            sh0,sh1,el0,el1,wr0,wr1 = arm_client.joint_states["frontleft"]
            arm_client.joint_move(sh0,sh1,el0,el1,wr0,wr1)

            input("Press any key to continue...")
            arm_client.carry_position()

            robot.logger.info("Moving arm to front")
            sh0,sh1,el0,el1,wr0,wr1 = arm_client.joint_states["frontright"]
            arm_client.joint_move(sh0,sh1,el0,el1,wr0,wr1)

            input("Press any key to continue...")

            '''
            state = robot_state_client.get_robot_state()
            joint_states = state.kinematic_state.joint_states
            for joint_state in joint_states:
                print(f"{joint_state.name}: {joint_state.position}")

            
            while True:
                delta = input("Twist shoulder by (radians, enter 'q' to exit): ")
                if delta == 'q':
                    break
                sh0 += float(delta)
                while sh0 > 2 * 3.14:
                    sh0 += -2 * 3.14
                arm_client.joint_move(sh0,sh1,el0,el1,wr0,wr1)
                state = robot_state_client.get_robot_state()
                joint_states = state.kinematic_state.joint_states
                for joint_state in joint_states:
                    print(f"{joint_state.name}: {joint_state.position}")

            while True:
                delta = input("Twist elbow by (radians, enter 'q' to exit): ")
                if delta == 'q':
                    break
                el1 += float(delta)
                while el1 > 2 * 3.14:
                    el1 += -2 * 3.14
                arm_client.joint_move(sh0,sh1,el0,el1,wr0,wr1)
                state = robot_state_client.get_robot_state()
                joint_states = state.kinematic_state.joint_states
                for joint_state in joint_states:
                    print(f"{joint_state.name}: {joint_state.position}")
            '''

            arm_client.carry_position()
            open_command = RobotCommandBuilder.claw_gripper_open_command()
            cmd_id = command_client.robot_command(open_command)
            robot.logger.info("Opening claw...")

            #stow arm again 
            arm_client.stow_arm()

            #dock
            #docking_client.dock(520)
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.exception("Threw an exception")
        return False


if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)