import urx
import time
from threading import Thread, Event
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
from urx.urscript import URScript
# from pynput import keyboard
# Initialize the robot by IP
rob = urx.Robot("192.168.51.254")

# Set the tool center point (TCP) and payload
rob.set_tcp((0, 0, 0.1, 0, 0, 0))
rob.set_payload(0.1, (0, 0, 0.1))

# Define motion parameters (acceleration and velocity)
a = 0.1  # Example acceleration
v = 0.1  # Example velocity

# Step sizes
step = 0.02
move_vector = [0, 0, 0]  # x, y, z

# Initialize a set to keep track of currently pressed keys
pressed_keys = set()

# Event to stop the robot motion thread
stop_event = Event()

def print_pose():
    pos = rob.get_pose().pos
    print(f"Current pose xyz is: {pos}")

def update_move_vector():
    global move_vector
    move_vector = [0, 0, 0]  # Reset move vector
    
    if 'w' in pressed_keys:
        move_vector[2] += step  # Up
    if 's' in pressed_keys:
        move_vector[2] -= step  # Down
    if 'a' in pressed_keys:
        move_vector[1] += step  # Left
    if 'd' in pressed_keys:
        move_vector[1] -= step  # Right
    if 'q' in pressed_keys:
        move_vector[0] += step  # Forward
    if 'e' in pressed_keys:
        move_vector[0] -= step  # Backward

def control_robot():
    while not stop_event.is_set():
        user_input = input("Enter direction keys (e.g., 'qweasd'): ").strip().lower()
        pressed_keys.clear()
        for char in user_input:
            if char in ['w', 's', 'a', 'd', 'q', 'e']:
                pressed_keys.add(char)


        update_move_vector()
        if move_vector != [0, 0, 0]:
            rob.movel(move_vector + [0, 0, 0], acc=a, vel=v, relative=True)
            print_pose()
        # time.sleep(0.1)  # Adjust the sleep time as needed


# 启动键盘监听器和机器人控制线程
if __name__ == "__main__":
    robot_thread = Thread(target=control_robot)
    robot_thread.start()

    # 使用try-except块以捕获键盘中断
    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()

    robot_thread.join()

