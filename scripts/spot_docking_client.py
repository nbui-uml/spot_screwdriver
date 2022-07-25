import argparse

import bosdyn.client
from bosdyn.client import robot_command
from bosdyn.client.docking import blocking_dock_robot, blocking_undock


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
        blocking_dock_robot(robot, dock_id)
        print("Docking success.")


    def undock(self) -> None:
        """
        Undocks the robot.
        """

        robot = self.robot
        assert not robot.is_estopped(), "Robot has been estopped."
        command_client = robot.ensure_client(robot_command.RobotCommandClient.default_service_name)

        robot.power_on()

        print("Undocking...")
        blocking_undock(robot)
        print("Undocked. Awaiting commands.")
        