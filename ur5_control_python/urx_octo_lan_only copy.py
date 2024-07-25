import cv2
import numpy as np
from octo.model.octo_model import OctoModel
import jax
import os
import urx
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
from urx.urscript import URScript
import time
# Initialize the robot by IP
rob = urx.Robot("192.168.51.254")
rob.set_tcp((0, 0, 0, 0, 0, 0))
rob.set_payload(0.1, (0, 0, 0.1))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small")
model = OctoModel.load_pretrained("/home/tong/model_space/7.24/model_8_octo1.5_win2_h4_10k/")
print('Successfully loaded the model!')

# Set window and camera
WINDOW_SIZE = 2
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

language_instruction = "move to the blue rubber duck"
task = model.create_tasks(texts=[language_instruction])  # For language conditioned

def capture_and_process_image():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        return None
    
    resized_frame = cv2.resize(frame, (256, 256))
    return resized_frame

def capture_images(window_size):
    images = []
    for _ in range(window_size):
        image = None
        while image is None:
            image = capture_and_process_image()
        images.append(image)
    return images

def display_images(images):
    combined_image = cv2.hconcat(images)
    cv2.imshow('frame', combined_image)
    cv2.waitKey(1)

def perform_action(images):
    input_images = np.stack(images)[None]
    print("input_images shape", input_images.shape)
    observation = { 
        'image_primary': input_images,
        'timestep_pad_mask': np.full((1, input_images.shape[1]), True, dtype=bool)
    }
    actions = model.sample_actions(
        observation, 
        task, 
        unnormalization_statistics=model.dataset_statistics["action"], 
        rng=jax.random.PRNGKey(0)
    )
    actions = actions[0]
    print("Pred_actions", actions[0])
    
    # Use URX to control robot
    rob.movel((actions[0][0], actions[0][1], actions[0][2], 0, 0, 0), acc=0.5, vel=0.5, relative=True)
    print("Current pose xyz is:", rob.get_pose().pos)
def main_loop():
    while True:
        images = capture_images(WINDOW_SIZE)
        display_images(images)
        perform_action(images)
        
        print('Waiting for the next images!')
        
        if cv2.waitKey(1) == 27:
            rob.close()
            break

# Move the robot to the initial position
rob.movej((-0.1554021 + 3.1415926 / 2, -1.6309904, 2.4249759, -2.406821, -1.5576328, 2.7911022), acc=1, vel=1)
print('Start to input 2 images')

main_loop()


