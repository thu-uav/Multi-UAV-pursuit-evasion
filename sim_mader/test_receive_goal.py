import rospy
from geometry_msgs.msg import PoseStamped

print("init receive call back node")
rospy.init_node("test_receive_goal")

class ReceiveCallBack:
    def __init__(self, idx) -> None:
        self.idx = idx
        self.subscriber = rospy.Subscriber(f"/SQ0{self.idx}s/term_goal", PoseStamped, self.call_back)
        pass
    
    def call_back(self, msg: PoseStamped):
        print(f"id = {self.idx} has received goal, goal pos = {msg.pose.position}")

for i in range(6):
    rcb = ReceiveCallBack(i)
    
while not rospy.is_shutdown():
    pass