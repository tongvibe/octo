import pygame
import urx
import time
from threading import Thread, Event
import cv2  # 用于视频采集
import numpy as np
import os
import pandas as pd  # 用于保存数据
from scipy.spatial.transform import Rotation as R

# 初始化 Pygame 和手柄
pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("请连接一个手柄并重试。")
    exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()

# 初始化机器人
rob = urx.Robot("192.168.51.254")

# 设置工具中心点 (TCP) 和载荷
rob.set_tcp((0, 0, 0., 0, 0, 0))
rob.set_payload(0.1, (0, 0, 0.1))

# 定义运动参数（加速度和速度）
a = 0.1  # 示例加速度
v = 0.1  # 示例速度

# 步长用于机器人移动
step = 0.02
rotation_step = 0.01 
move_vector = [0, 0, 0, 0, 0, 0]  # x, y, z, Rx, Ry, Rz

# 事件用于停止机器人运动线程
stop_event = Event()

# 打开摄像头
cap = cv2.VideoCapture(3)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 获取或创建下一个episode的文件夹
def get_next_episode_dir(base_path="training_data"):
    os.makedirs(base_path, exist_ok=True)
    episode_num = 0
    while os.path.exists(os.path.join(base_path, f"episode_{episode_num}")):
        episode_num += 1
    episode_dir = os.path.join(base_path, f"episode_{episode_num}")
    image_dir = os.path.join(episode_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    return episode_dir, image_dir

# 获取下一个episode的目录
episode_dir, image_dir = get_next_episode_dir()
csv_path = os.path.join(episode_dir, "data.csv")

# 用于数据存储的队列
data_queue = []

def print_pose():
    pos = rob.get_pose()
    print(f"\n Current pose xyzypr is: {pos}")

def update_move_vector(x, y, z, rx, ry, rz):
    global move_vector
    move_vector = [x * step, y * step, z * step, rx * rotation_step, ry * rotation_step, rz * rotation_step]

def control_robot():
    while not stop_event.is_set():
        if move_vector != [0, 0, 0, 0, 0, 0]:
            rob.movel(move_vector, acc=a, vel=v, relative=True)
            print_pose()
        time.sleep(0.1)  # 根据需要调整睡眠时间

def handle_joystick():
    while not stop_event.is_set():
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                axis_x = joystick.get_axis(0)  # X轴
                axis_y = joystick.get_axis(1)  # Y轴
                axis_z = 0  # 0-a, 1-b, 3-x, 4-y 10-select, 11-start

                axis_rx = -joystick.get_axis(3)  # RX轴（假设手柄有 RX 轴）
                axis_ry = joystick.get_axis(2)  # RY轴（假设手柄有 RY 轴）
                axis_rz = 0  # RZ轴

                update_move_vector(-axis_x, axis_y, -axis_z, axis_rx, axis_ry, axis_rz)
            ##!!!!!!!!这里可以优化一下，xyz一起动，这里分开更新vector了，一起DONG
            elif event.type == pygame.JOYHATMOTION:
                axis_z = joystick.get_hat(0)
                if axis_z == (0, 1):
                    z = 1
                elif axis_z == (0, -1):
                    z = -1
                else:
                    z = 0
                update_move_vector(move_vector[0], move_vector[1], z, move_vector[3], move_vector[4], move_vector[5])
            elif event.type == pygame.JOYBUTTONDOWN or event.type == pygame.JOYBUTTONUP:
                if joystick.get_button(0):
                    axis_rz = 1
                elif joystick.get_button(4):
                    axis_rz = -1
                elif joystick.get_button(1):
                    axis_rz = 0
                    save_data()
                    stop_event.set()
                    rob.close()
                    cap.release()
                    exit()
                else:
                    axis_rz = 0
                update_move_vector(move_vector[0], move_vector[1], move_vector[2], move_vector[3], move_vector[4], axis_rz)
            
        # time.sleep(0.1)  # 调整以满足需要的刷新速率

def handle_joystick():
    while not stop_event.is_set():
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                axis_x = joystick.get_axis(0)  # X轴
                axis_y = joystick.get_axis(1)  # Y轴
                # axis_z = 0  # 0-a, 1-b, 3-x, 4-y 10-select, 11-start

                axis_rx = -joystick.get_axis(3)  # RX轴（假设手柄有 RX 轴）
                axis_ry = joystick.get_axis(2)  # RY轴（假设手柄有 RY 轴）
                axis_rz = 0  # RZ轴

                # update_move_vector(-axis_x, axis_y, -axis_z, axis_rx, axis_ry, axis_rz)
            ##!!!!!!!!这里可以优化一下，xyz一起动，这里分开更新vector了，一起DONG
                if event.type == pygame.JOYHATMOTION:
                    axis_z = joystick.get_hat(0)
                    if axis_z == (0, 1):
                        z = 1
                    elif axis_z == (0, -1):
                        z = -1
                    else:
                        z = 0
                    # update_move_vector(move_vector[0], move_vector[1], z, move_vector[3], move_vector[4], move_vector[5])
                elif event.type == pygame.JOYBUTTONDOWN or event.type == pygame.JOYBUTTONUP:
                    if joystick.get_button(0):
                        axis_rz = 1
                    elif joystick.get_button(4):
                        axis_rz = -1
                    elif joystick.get_button(1):
                        axis_rz = 0
                        save_data()
                        stop_event.set()
                        rob.close()
                        cap.release()
                        exit()
                    else:
                        axis_rz = 0

                    
                update_move_vector(-axis_x, axis_y, z, axis_rx, axis_ry, axis_rz)

def convert_to_euler_angles(rotation_vector):
    """将旋转向量(Rx, Ry, Rz)转换为欧拉角(roll, pitch, yaw)"""
    rotation = R.from_rotvec(rotation_vector)
    euler_angles = rotation.as_euler('xyz', degrees=False)
    return euler_angles

def collect_data():
    image_counter = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            joints_space = rob.getj()
            csys_pose = rob.getl()
            rotation_action = move_vector
            # 提取旋转向量部分 [Rx, Ry, Rz]
            rotation_vector = move_vector[3:6]
            rotation = R.from_rotvec(rotation_vector)
            # 将旋转对象转换为欧拉角
            euler_angles = rotation.as_euler('xyz', degrees=False)
            # 更新 action 的后三个值为欧拉角
            euler_action = move_vector[0:3] + euler_angles.tolist()
            timestamp = time.time()
            image_filename = os.path.join(image_dir, f"image_{image_counter}.png")
            data = {
                'timestamp': timestamp,
                'joints_space': joints_space,
                'csys_pose': csys_pose,
                'rotation_action': rotation_action,
                'euler_action': euler_action,
                'image_path': image_filename
            }
            data_queue.append(data)
            # Save the image
            cv2.imwrite(image_filename, frame)
            image_counter += 1

        time.sleep(0.25)  # 根据需要调整采集频率

def save_data():
    # 将数据保存为CSV文件
    df = pd.DataFrame(data_queue)
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")


def main():
    # 创建并启动机器人运动线程
    robot_thread = Thread(target=control_robot)
    robot_thread.start()

    # 启动手柄输入监听线程
    joystick_thread = Thread(target=handle_joystick)
    joystick_thread.start()

    # 启动数据采集线程
    data_thread = Thread(target=collect_data)
    data_thread.start()

    try:
        robot_thread.join()
        joystick_thread.join()
        data_thread.join()
    except KeyboardInterrupt:
        stop_event.set()
        robot_thread.join()
        joystick_thread.join()
        data_thread.join()

    # save_data()
    # rob.close()
    # cap.release()

if __name__ == "__main__":
    main()