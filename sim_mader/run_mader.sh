sh kill_mader.sh
roslaunch mader mader_general.launch
$ISAACSIM2_PATH/python.sh start_command.py
$ISAACSIM2_PATH/python.sh start_mader.py
$ISAACSIM_PATH/python.sh send_goal.py
$ISAACSIM2_PATH/python.sh sim_0907.py