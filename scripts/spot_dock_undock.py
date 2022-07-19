import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.client import robot_command
from bosdyn.client.docking import blocking_dock_robot, blocking_undock

class DockingClient:
    def __init__(self, config):
        bosdyn.client.util.setup_logging(config.verbose)
        self.sdk = bosdyn.client.create_standard_sdk("DockingClient")
        self.robot = self.sdk.create_robot(config.hostname)
        bosdyn.client.util.authenticate(self.robot)
        self.robot.time_sync.wait_for_sync() 

    def dock(self, dock_id):
        robot = self.robot

        command_client = robot.ensure_client(robot_command.RobotCommandClient.default_service_name)
        robot.power_on()
        robot_command.blocking_stand(command_client)

        print("Docking")
        blocking_dock_robot(robot, dock_id)
        print("Docking success")

    def undock(self):
        robot = self.robot
        
        command_client = robot.ensure_client(robot_command.RobotCommandClient.default_service_name)
        robot.power_on()

        print("Undocking")
        blocking_undock(robot)
        print("Undocked. Awaiting commands.")
        