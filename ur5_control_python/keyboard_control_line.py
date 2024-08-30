import urx
import time
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
from urx.urscript import URScript
from pynput import keyboard
# Initialize the robot by IP
rob = urx.Robot("192.168.51.254")

# Set the tool center point (TCP) and payload
# set the offset and load for robot

rob.set_tcp((0, 0, 0.1, 0, 0, 0))
rob.set_payload(2, (0, 0, 0.1))

# Allow some time for the robot to process the setup commands
time.sleep(0.2)

# Define motion parameters (acceleration and velocity)
a = 0.05  # example acceleration
v = 0.05   # example velocity

def on_press(key):
    try:
        if key.char == 'w':  # 键盘上的 '8' 键，上移
            rob.movel((0, 0, 0.02, 0, 0, 0), acc=a, vel=v, relative=True)
            print(" --> Current pose xyz is:", rob.get_pose().pos)
        elif key.char == 's':  # 键盘上的 '2' 键，下移
            rob.movel((0, 0, -0.02, 0, 0, 0), acc=a, vel=v, relative=True)
            print(" --> Current pose xyz is:", rob.get_pose().pos)
        elif key.char == 'a':  # 键盘上的 '4' 键，向左移动
            rob.movel((0, 0.02, 0, 0, 0, 0), acc=a, vel=v, relative=True)
            print(" --> Current pose xyz is:", rob.get_pose().pos)
        elif key.char == 'd':  # 键盘上的 '6' 键，向右移动
            rob.movel((0, -0.02, 0, 0, 0, 0), acc=a, vel=v, relative=True)
            print(" --> Current pose xyz is:", rob.get_pose().pos)
        elif key.char == 'q':  # 键盘上的 '6' 键，向前移动
            rob.movel((0.02, 0, 0, 0, 0, 0), acc=a, vel=v, relative=True)
            print(" --> Current pose xyz is:", rob.get_pose().pos)
        elif key.char == 'e':  # 键盘上的 '6' 键，向后移动
            rob.movel((-0.02, 0, 0, 0, 0, 0), acc=a, vel=v, relative=True)
            print(" --> Current pose xyz is:", rob.get_pose().pos)
    except AttributeError:
        pass

def on_release(key):
    if key == keyboard.Key.esc:
        # 停止监听
        rob.close()
        return False

# 启动键盘监听器
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()




