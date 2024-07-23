import cv2
import numpy as np
from octo.octo.model.octo_model import OctoModel
import jax
import tensorflow_datasets as tfds
import os
import time
import urx
import time
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
from urx.urscript import URScript

# Initialize the robot by IP
rob = urx.Robot("192.168.51.254")

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small")

model = OctoModel.load_pretrained("/home/tong/robotic-ai/octo/octo-base")
print('Successfully load the model!')

def pad_to_square_opencv(image):
    # 获取图像的宽度和高度
    height, width, _ = image.shape
    
    # 计算填充后的正方形边长
    max_side = max(width, height)
    
    # 创建一个新的正方形图像，背景为白色
    new_image = 255 * np.ones((max_side, max_side, 3), dtype=np.uint8)
    
    # 计算图像在新正方形图像中的位置
    left = (max_side - width) // 2
    top = (max_side - height) // 2
    
    # 将原图像粘贴到新图像上
    new_image[top:top+height, left:left+width] = image
    
    return new_image

#set window and camera
WINDOW_SIZE = 2
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_BUFFERSIZE, WINDOW_SIZE) 
#my camera port '2'
# width = 1280  # 设置宽度为 640 像素
# height = 720  # 设置高度为 480 像素
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

images = []
# language_instruction="sweep the green cloth to the left side of the table"
goal_image = cv2.imread('/home/tong/robotic-ai/ur5_control_python/goal_img1.jpg')
goal_image = cv2.cvtColor(goal_image, cv2.COLOR_BGR2RGB)
goal_image = cv2.resize(goal_image, (256, 256))
task = model.create_tasks(goals={"image_primary": goal_image[None]})   # for goal-conditioned
# task = model.create_tasks(texts=[language_instruction])                  # for language conditioned

# 进行推理循环
pred_actions = []

rob.movej((-3.2969947,-1.6309904,2.4249759,-2.406821,-1.5576328,2.7911022))
print('Start to input 2 images')
while True:
        ret, frame = cap.read()
        # print("frame shape",frame.shape)
        # cv2.imshow('frame', frame)
        
        # if cv2.waitKey(1) == 32:
       
            
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 从可能的 BGR 格式开始
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式进行某些处理

        # 现在你想显示这个图像，用 OpenCV 显示，因此需要转换回 BGR
        resized_frame_rgb = cv2.resize(frame_rgb, (256, 256))
        resized_frame_bgr = cv2.cvtColor(resized_frame_rgb, cv2.COLOR_RGB2BGR)  # 转换回 BGR 格式

        # 使用 OpenCV 显示这个 BGR 格式的图像
        cv2.imshow('frame', resized_frame_bgr)
        # square_rgb_frame=pad_to_square_opencv(frame_rgb)

        # resized_frame = cv2.resize(frame_rgb, (256,256))
        # cv2.imshow('frame', frame)
        # time.sleep(1)
        images.append(resized_frame_rgb)
        
        # if len(images) > WINDOW_SIZE:
        #     images.pop(0)
        if len(images) == WINDOW_SIZE:
            input_images = np.stack(images)[None]
            print("input_images shape",input_images.shape)
            observation = { 
                'image_primary': input_images,
                'pad_mask': np.full((1, input_images.shape[1]), True, dtype=bool)
            }

            # 采样动作
            norm_actions = model.sample_actions(observation, task, rng=jax.random.PRNGKey(0))
            log = model.get_pretty_spec()
            print(log)
            norm_actions = norm_actions[0]  # 移除批次维度
            actions = (
                norm_actions * model.dataset_statistics["berkeley_autolab_ur5"]['action']['std']
                + model.dataset_statistics["berkeley_autolab_ur5"]['action']['mean']
            )
            images.pop(0)
            
            print("Pred_actions",actions[0])
            print('Waiting 1s for the next image!')
            pred_actions.append(actions[0])
            # time.sleep(3)
            # cv2.imshow('frame', resized_frame)

            #use urx control robot
            rob.movel((actions[0][1], actions[0][0], actions[0][2], 0,0,0), acc=0.1, vel=0.1, relative=True)
            # print(delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw)
            time.sleep(0.02)
            print("Current pose xyz is:", rob.get_pose().pos)   


            if cv2.waitKey(1) == 27:
                rob.close()
                break
                
                # print(i)
            else:
                print("############################################################################################################################")
            # with open('q_targets.txt', 'w') as f:
            #     for arr in q_total:
            #         line = ', '.join(map(str, arr))
            #         f.write(f'{line}\n')
            # 显示摄像头画面
            # cv2.imshow('frame', resized_frame)
            # time.sleep(0.5)
        
        
# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
rob.close()





                
            


