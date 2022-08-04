import argparse

import bosdyn.client
from bosdyn.client import robot_command
from bosdyn.client import docking


class DockingClient:
    def __init__(self, config: argparse.Namespace, robot: bosdyn.client.Robot) -> None:
        """
        Creates a DockingClient instance.

        Parameters
        -----
        config: Namespace
        robot: Robot
            The robot instance.
        """
        self.sdk = bosdyn.client.create_standard_sdk("DockingClient")
        self.robot = robot


    def dock(self, dock_id: int) -> None:
        """
        Commands the robot to dock at the location specified by dock_id.
        Parameters
        -----
        dock_id: int 
            ID of the dock to go to.
        """

        robot = self.robot
        assert not robot.is_estopped(), "Robot has been estopped."
        command_client = robot.ensure_client(robot_command.RobotCommandClient.default_service_name)
        
        robot.power_on()
        robot_command.blocking_stand(command_client)

        print("Docking...")
        docking.blocking_dock_robot(robot, dock_id)
        print("Docking success.")
        

    def undock(self) -> bool:
        """
        Undocks the robot if docked and returns True. Otherwise, does nothing and returns False.
        """

        robot = self.robot
        assert not robot.is_estopped(), "Robot has been estopped."
        docking_client = robot.ensure_client(docking.DockingClient.default_service_name)

        docking_state = docking_client.get_docking_state()

        if(docking_state.status == 1):
            robot.power_on()

            print("Undocking...")
            docking.blocking_undock(robot)
            print("Undocked. Awaiting commands.")
            return True
        return False


    def start(self):
        """
        Starts the robot by turning on the motors and undocks/stands the robot.
        """
        robot = self.robot
        assert not robot.is_estopped(), "Robot has been estopped."

        if(self.undock() == False):
            if not robot.is_powered_on():
                robot.logger.info("Powering on robot... This may take a several seconds.")
                robot.power_on(timeout_sec=20)
                assert robot.is_powered_on(), "Robot power on failed."
                robot.logger.info("Robot powered on.")

            robot.logger.info("Commanding robot to stand...")
            command_client = robot.ensure_client(robot_command.RobotCommandClient.default_service_name)
            robot_command.blocking_stand(command_client, timeout_sec=10)
            robot.logger.info("Robot standing.")
        