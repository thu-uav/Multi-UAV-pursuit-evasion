import argparse
import os
import time

def convertToStringCommand(quad):
    return "roslaunch mader mader.launch quad:="+quad

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
    session_name = "mader_session"
    commands = []
    for i in range(1, n+1):
        command = convertToStringCommand(quad='SQ0'+ str(i-1) + 's')
        commands.append(command)
    
    create_session(session_name, commands)
    time.sleep(n)
    # os.system("tmux attach")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drone-num", type=int, default=6)
    args = parser.parse_args()
    main(args)