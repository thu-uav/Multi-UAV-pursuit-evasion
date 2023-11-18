#!/home/zanghongzhi/.local/share/ov/pkg/isaac_sim-2022.2.0/python.sh
import argparse
import os
import time
import sys

dir_name = os.path.dirname(os.path.abspath(__file__))
OMNIDRONES_ENV_DIR = os.path.dirname(dir_name)
sys.path.append(OMNIDRONES_ENV_DIR)

# from omni_drones.envs.formation_ball_copy import FormationBallCopy, REGULAR_TETRAGON

# from test_multi_ball import FORMATION
REGULAR_TETRAGON = [
    [0, 0, 0],
    [2, 2, 0],
    [2, -2, 0],
    [-2, -2, 0],
    [-2, 2, 0],
]

REGULAR_PENTAGON = [
    [2., 0, 0],
    [0.618, 1.9021, 0],
    [-1.618, 1.1756, 0],
    [-1.618, -1.1756, 0],
    [0.618, -1.9021, 0],
    [0, 0, 0]
]

TARGET_CENTER = [0.0, 20.0, 1.5]
FORMATION = REGULAR_PENTAGON

GOAL = []
for i in range(1, len(FORMATION)+1):
    goal_i = [FORMATION[i-1][0] + TARGET_CENTER[0], FORMATION[i-1][1] +TARGET_CENTER[1], FORMATION[i-1][2] + TARGET_CENTER[2]]
    GOAL.append(goal_i)

    
def convertToStringCommand(quad, goal):
    return "rostopic pub /"+quad+"/term_goal geometry_msgs/PoseStamped '{header: {stamp: now, frame_id: 'world'}, pose: {position: {x: "+str(goal[0])+", y: "+str(goal[1])+", z: "+str(goal[2])+"}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 0.0}}}'"


def create_session(session_name, commands):
    os.system("tmux new -d -s "+str(session_name)+" -x 300 -y 300")
    for i in range(len(commands)):
        print('splitting ',i)
        os.system('tmux split-window ; tmux select-layout tiled')
   
    for i in range(len(commands)):
        os.system('tmux send-keys -t '+str(session_name)+':0.'+str(i) +' "'+ commands[i]+'" '+' C-m')
    print("Commands sent")


def main(args):
    # pass
    n = args.drone_num
    session_name = "send_goal_session"
    goal_list = []
    commands = []
    for i in range(1, n+1):
        # goal_i = [0., 2., 1.5]        
        # goal_i = [FORMATION[i-1][0] + target_center[0], FORMATION[i-1][1] + target_center[1], FORMATION[i-1][2] + target_center[2]]
        # goal_list.append(goal_i)
        goal_i = GOAL[i-1]
        command = convertToStringCommand(quad='SQ0'+ str(i-1) + 's', goal=goal_i)
        commands.append(command)
    
    create_session(session_name, commands)
    # for command in commands:
    #     print(command)
    #     os.system(command)
    time.sleep(n)
    # os.system("tmux kill-session -t" + session_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drone-num", type=int, default=6)
    args = parser.parse_args()
    main(args)