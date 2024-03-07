import networktables


class NetworkCommunication:
    def __init__(self):
        """
        NetworkCommunications is a class allowing the communication of robot data between the roborio and coproccesser
        via networktables
        """
        self.ntinst = networktables.NetworkTablesInstance.getDefault()
        self.ntinst.startClientTeam(8775)
        self.ntinst.startDSClient()
        self.rotations_table = self.ntinst.getTable("Rotations")

    def send_pose(self, angle: float):
        self.rotations_table.putNumberArray("rot", angle)