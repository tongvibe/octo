import urx
import time
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
from urx.urscript import URScript
from scipy.spatial.transform import Rotation as R  # 导入scipy库中的Rotation类

# Initialize the robot by IP
rob = urx.Robot("192.168.51.254")

# Set the tool center point (TCP) and payload
# set the offset and load for robot

rob.set_tcp((0, 0, 0, 0, 0, 0))
rob.set_payload(1, (0, 0, 0.9))

# Allow some time for the robot to process the setup commands
time.sleep(0.2)

# Define motion parameters (acceleration and velocity)
a = 0.05  # example acceleration
v = 0.05   # example velocity

# Define position/orientation parameters
# x, y, z = 0.05, 0.05, 0.05  # example positions
# rx, ry, rz = 0, 0, 0  # example orientations

try:
    # Move to a joint space position
    # rob.movej((1, 2, 3, 4, 5, 6), acc=a, vel=v)

    # Move to a joint space position
    # rob.movej((-3.2969947,-1.6309904,2.4249759,-2.406821,-1.5576328,2.7911022))

    # Print the current tool pose
    print("Current tool pose is: ", rob.getl())
    print("Current pose xyz is:", rob.get_pose())
    print("Jointspace:", rob.getj())
    print(rob.get_pos())
    trans = rob.get_pose()  # get current transformation matrix (tool to base)
    pos = trans.pos._data  # [x, y, z]
    rotation = R.from_rotvec(trans.orient.rv._data )
    euler_angles = rotation.as_euler('xyz', degrees=False)
    print("欧拉角:", euler_angles)

    # Move relative to the current pose
    rob.movel((-0.58479328, -0.46829172 , 0.163,  -0.41, -0.372,  0.139), acc=a, vel=v, relative=False)
    # txt_file = '/home/tong/robotic-ai/octo/examples/action_groundturth.txt'
    # with open(txt_file, 'r') as f:
    #     # 逐行读取数据
    #     for line in f:
    #         # 分割每行数据，假设数据以空格或制表符分隔
    #         data = line.strip().split(',')
            
    #         # 将数据转换为浮点数，并添加到对应的列表中
    #         delta_x = (float(data[0]))
    #         delta_y = (float(data[1]))
    #         delta_z = (float(data[2]))
    #         delta_roll = (float(data[3]))
    #         delta_pitch = (float(data[4]))
    #         delta_yaw = (float(data[5]))

    #         rob.movel((delta_y, delta_x, -delta_z, 0,0,0), acc=a, vel=v, relative=True)
    #         print(delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw)
    #         time.sleep(0.02)
            # print("Current pose xyz is:", rob.get_pose().pos)   
    # Translate the tool while keeping orientation
    # rob.translate((0.1, 0, 0), acc=a, vel=v)

    # Stop the robot with joint deceleration
    # rob.stopj(acc=a)

    # Move without waiting for completion
    # rob.movel((x, y, z, rx, ry, rz), wait=False)

    # while True:
    #     time.sleep(0.1)  # Sleep first, as the robot may not have processed the command yet
    #     if rob.is_program_running():
    #         break

    # # Move without waiting and monitor force
    # # rob.movel((x, y, z, rx, ry, rz), wait=False)
    # while rob.get_force() < 50:
    #     time.sleep(0.01)
    #     if not rob.is_program_running():
    #         break

    # # Stop the robot with linear deceleration
    # rob.stopl()

    # Example of error handling during movement
    # try:
        # rob.movel((0, 0, 0.1, 0, 0, 0), relative=True)
except Exception as e:
    print("Shit:", e)
    rob.close()

# finally:
# #     # Always safely close the connection to the robot
#     rob.close()


##Current tool pose is:  [-0.433010391343245, -0.05908719161139308, 0.11103314562850851, -2.1117846593432494, -2.1544580360293537, -0.0028301757725795722]

