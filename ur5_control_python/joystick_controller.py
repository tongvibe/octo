import pygame
import urx
import time
from threading import Thread, Event

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
rob.set_tcp((0, 0, 0.1, 0, 0, 0))
rob.set_payload(0.1, (0, 0, 0.1))

# 允许一些时间让机器人处理设置命令
# time.sleep(0.2)

# 定义运动参数（加速度和速度）
a = 0.1  # 示例加速度
v = 0.1  # 示例速度

# 步长用于机器人移动
step = 0.02
rotation_step = 0.01 
move_vector = [0, 0, 0, 0, 0, 0]  # x, y, z, Rx, Ry, Rz

# 事件用于停止机器人运动线程
stop_event = Event()

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
        # time.sleep(0.1)  # 根据需要调整睡眠时间

def handle_joystick():
    while not stop_event.is_set():
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                axis_x = joystick.get_axis(0)  # X轴
                axis_y = joystick.get_axis(1)  # Y轴
                axis_z = 0 #0-a, 1-b, 3-x, 4-y 10-select, 11-start

                #!!!!!!!
                axis_rx = -joystick.get_axis(3)# RX轴（假设手柄有 RX 轴）
                axis_ry = joystick.get_axis(2)  # RY轴（假设手柄有 RY 轴）
                axis_rz = 0  # RZ轴---> YA 

                update_move_vector(-axis_x, axis_y, -axis_z, axis_rx, axis_ry, axis_rz)
            elif event.type == pygame.JOYHATMOTION:
                axis_z = joystick.get_hat(0)
                if axis_z == (0, 1):
                    z = 1
                elif axis_z == (0, -1):
                    z = -1
                else:
                    z =0
                update_move_vector(move_vector[0], move_vector[1], z, move_vector[3], move_vector[4], move_vector[5])
            elif event.type == pygame.JOYBUTTONDOWN or event.type == pygame.JOYBUTTONUP:
                if joystick.get_button(0):
                    axis_rz = 1
                elif joystick.get_button(4):
                    axis_rz = -1
                elif joystick.get_button(3):
                    axis_rz = 0
                    rob.movej((-0.1554021+3.1415926/2,-1.6309904,2.4249759,-2.406821,-1.5576328,2.7911022),acc = 1, vel = 1)
                    print('Return to home!! Please restart the code!!!!!!!')
                    stop_event.set()
                    rob.close()
                    exit()
                 
                else:
                    axis_rz = 0
                update_move_vector(move_vector[0], move_vector[1], move_vector[2], move_vector[3], move_vector[4], axis_rz)

                


                # z = 0 #

        # time.sleep(0.1)  # 调整以满足需要的刷新速率

def main():
    # 创建并启动机器人控制线程
    robot_thread = Thread(target=control_robot)
    robot_thread.start()

    # 启动手柄输入监听线程
    joystick_thread = Thread(target=handle_joystick)
    joystick_thread.start()

    robot_thread.join()
    joystick_thread.join()

    rob.close()

if __name__ == "__main__":
    main()