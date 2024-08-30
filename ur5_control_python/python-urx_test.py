import urx
import time
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
from urx.urscript import URScript
from scipy.spatial.transform import Rotation as R  # 导入scipy库中的Rotation类
import numpy as np
import math
# Initialize the robot by IP
rob = urx.Robot("192.168.51.254")

# # Set the tool center point (TCP) and payload
# # set the offset and load for robot

rob.set_tcp((0, 0, 0.09, 0, 0, 0))
# rob.set_payload(1, (0, 0, 0.9))

# Allow some time for the robot to process the setup commands
time.sleep(0.2)

# Define motion parameters (acceleration and velocity)
a = 0.05  # example acceleration
v = 0.05   # example velocity
#计算

def modify_rotation_in_transformation_matrix(T, R):
    """
    修改齐次变换矩阵中的旋转矩阵。

    :param T: 原始齐次变换矩阵 (4x4)
    :param R: 要乘以的旋转矩阵 (3x3)
    :return: 修改后的齐次变换矩阵 (4x4)
    """

    # 先确保输入的矩阵维度正确
    assert T.shape == (4, 4), "T must be a 4x4 matrix"
    assert R.shape == (3, 3), "R must be a 3x3 matrix"

    # 提取原有的旋转部分（前3x3）
    original_rotation = T[:3, :3]

    # 计算新的旋转矩阵
    new_rotation = np.dot(R, original_rotation)

    # 将新的旋转矩阵放回齐次变换矩阵中
    T[:3, :3] = new_rotation

    return T


cam_T_bottle = np.array(
    [[0.056892, -0.987741, -0.145368, 0.037548],
    [-0.439035, 0.106018, -0.892193, -0.171583],
    [0.896667, 0.114581, -0.427621, 0.745482],
    [0.000000, 0.000000, 0.000000, 1.000000]]
    )




# Tz = np.array(
#     [[-1,0,0,0],
#      [0,-1,0,0],
#      [0,0,1,0],
#      [0,0,0,1]]
# )
# Ty = np.array(
#     [[0,0,-1],
#      [0,1,0],
#      [1,0,0]]
# )

T = np.array(
    [[0,0,1],
     [1,0,0],
     [0,1,0]]
)
# [0,0,1;
# 1,0,0;
# 0,1,0]

corrected_matrix = modify_rotation_in_transformation_matrix(cam_T_bottle, T)
# cam_T_bottle = np.array(
#     [[-0.896667, -0.114581 , 0.427621, 0.037548],
#     [-0.439035 , 0.106018, -0.892193, -0.171583],
#     [ 0.056892 ,-0.987741 ,-0.145368, 0.745482],
#     [0.000000, 0.000000, 0.000000, 1.000000]]

#     )


# cam_T_bottle = np.dot(Tnb, cam_T_bottle)
# cam_T_bottle = np.dot(Tx, cam_T_bottle)
# cam_T_bottle = np.dot(Ty, cam_T_bottle)
Y = np.array(
    [[-0.99691752, -0.04396394, -0.06498177, -0.335 ],  #-0.3301283   -0.35202257
    [ 0.00679575,  0.77674946, -0.62977305,  0.04730107],###0.05730
    [ 0.07816186, -0.62827339, -0.77405637,  0.66619011],
    [ 0.    ,      0.     ,     0.     ,     1.        ]]    
)
base_T_tag = np.dot(Y, corrected_matrix)

# base_T_tag = np.array(
#     [[0.06245558, 0.99176524 , -0.11180769, -0.37611806],
#     [0.82147682 , -0.11470462,  -0.55858639, -0.26322131],
#     [0.56681146, 0.05696057, 0.82187608,  0.04590873],
#     [ 0.      ,    0.        ,  0.      ,    1.        ]]
#                 )

print("base_T_tag",base_T_tag)
# Define position/orientation parameters
# x, y, z = 0.05, 0.05, 0.05  # example positions
# rx, ry, rz = 0, 0, 0  # example orientations
def homogeneous_to_rot_trans(T):
    # 提取平移向量
    tx, ty, tz = T[0, 3], T[1, 3], T[2, 3]
    # 提取旋转矩阵
    R = T[0:3, 0:3]
    # 计算旋转角度 θ
    trace_R = np.trace(R)
    theta = math.acos((trace_R - 1) / 2)
    if theta < 1e-6:
        # 如果旋转角度接近0，旋转向量为零向量S
        return np.array([0, 0, 0]), np.array([tx, ty, tz])
    # 计算旋转向量
    rx = (R[2, 1] - R[1, 2]) / (2 * math.sin(theta))
    ry = (R[0, 2] - R[2, 0]) / (2 * math.sin(theta))
    rz = (R[1, 0] - R[0, 1]) / (2 * math.sin(theta))
    rot_vector = theta * np.array([rx, ry, rz])
    # print(tx,ty,tz,rot_vector)
    return np.append(np.array([tx, ty, tz]),rot_vector)

ur5_order = homogeneous_to_rot_trans(base_T_tag)
print(ur5_order)

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
    # rob.movel((ur5_order), acc=a, vel=v, relative=False)
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

