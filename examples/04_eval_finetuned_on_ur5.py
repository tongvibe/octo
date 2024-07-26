"""
This script shows how to evaluate a finetuned model on a real UR5 robot.
How to use it:
python examples/04_eval_finetuned_on_ur5.py --checkpoint_weights_path=/home/tong/model_space/7.24/model_8_octo1.5_win2_h4_10k/ --checkpoint_step=9999 --im_size=256
"""

from datetime import datetime
from functools import partial
import os
import time

from absl import app, flags, logging
import click
import cv2
import imageio
import jax
import jax.numpy as jnp
import numpy as np
import urx
import gym
from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, TemporalEnsembleWrapper
from octo.utils.train_callbacks import supply_rng

np.set_printoptions(suppress=True)
logging.set_verbosity(logging.ERROR)

FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_weights_path", None, "Path to checkpoint", required=True)
flags.DEFINE_integer("checkpoint_step", None, "Checkpoint step", required=True)

# Custom to UR5 robot
flags.DEFINE_string("ip", "192.168.51.254", "IP address of the robot")
flags.DEFINE_float("acc", 0.5, "Acceleration for robot movements")
flags.DEFINE_float("vel", 0.5, "Velocity for robot movements")
flags.DEFINE_spaceseplist("goal_pose", [0.5, 0.0, 0.3], "Goal position and orientation")
flags.DEFINE_spaceseplist("initial_pose", [0.5, 0.0, 0.3], "Initial position and orientation")
flags.DEFINE_bool("blocking", False, "Use the blocking controller")
flags.DEFINE_integer("im_size", None, "Image size", required=True)
flags.DEFINE_string("video_save_path", None, "Path to save video")
flags.DEFINE_integer("num_timesteps", 50, "Number of timesteps")
flags.DEFINE_integer("window_size", 2, "Observation history length")
flags.DEFINE_integer("action_horizon", 4, "Length of action sequence to execute/ensemble")
flags.DEFINE_bool("show_image", False, "Show image")

STEP_DURATION = 0.2



def null_obs(img_size):
    return {
        "image_primary": np.zeros((img_size, img_size, 3), dtype=np.uint8),
        "proprio": np.zeros((6,), dtype=np.float64)
    }

def convert_obs(pose, image, im_size):
    image_obs = (image.reshape(im_size, im_size, 3)).astype(np.uint8)
    # cv2.imshow('imagw',image_obs)
    proprio = pose
    return {
        "image_primary": image_obs,
        # "proprio": proprio,
    }

class UR5Env(gym.Env):
    def __init__(self, ip, camera_ID=0, im_size: int=256, acc=0.5, vel=0.5):
        self.rob = urx.Robot(ip)
        self.cap = cv2.VideoCapture(camera_ID) 
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.rob.set_tcp((0, 0, 0, 0, 0, 0))
        self.rob.set_payload(0.1, (0, 0, 0.1))
        self.acc = acc
        self.vel = vel
        self.im_size = im_size
        self.observation_space = gym.spaces.Dict(
            {
                "image_primary": gym.spaces.Box(
                    low=np.zeros((int(im_size), int(im_size), 3)),
                    high=255 * np.ones((int(im_size), int(im_size), 3)),
                    dtype=np.uint8,
                ),
                "proprio": gym.spaces.Box(
                    low=np.ones((6,)) * -1, high=np.ones((6,)), dtype=np.float64
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=np.zeros((6,)), high=np.ones((6,)), dtype=np.float64
        )

    def step(self, action):
        self.rob.movel((action[0], action[1], action[2], 0, 0, 0), acc=self.acc, vel=self.vel, relative=True)
        obs = self.get_observation()
        reward = self.compute_reward(obs)
        done = self.check_done(obs)
        truncated = False
        return obs, reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        initial_pose = [-0.1554021 + 3.1415926 / 2, -1.6309904, 2.4249759, -2.406821, -1.5576328, 2.7911022]
        self.rob.movej(initial_pose, acc=self.acc, vel=self.vel)
        return self.get_observation(), {}

    def get_observation(self):
        pose = self.rob.get_pose().pos
        image = self.get_image()
        return convert_obs(pose, image, self.im_size)

    def compute_reward(self, obs):
        # Implement your reward function
        return 0

    def check_done(self, obs):
        # Implement your logic to check if the task is done
        return False

    def get_image(self):
        # Implement image capture logic, e.g., from a camera
        # Return a dummy image for now
        ret, frame = self.cap.read()  # 从摄像头获取一帧图像
        if ret:  # 如果成功获取到图像
            frame = cv2.resize(frame, (self.im_size, self.im_size))  # 调整图像尺寸为 im_size x im_size
            # cv2.imshow('imagw',frame)
            return frame
        else:  # 如果未能成功获取图像
            return None  # 或者采取其他处理措施
        # return np.zeros((self.im_size, self.im_size, 3), dtype=np.uint8)

    def close(self):
        self.cap.release()
        self.rob.close()


def main(_):
    env = UR5Env(ip = FLAGS.ip, camera_ID=2, acc = FLAGS.acc, vel = FLAGS.vel)
    env = HistoryWrapper(env, FLAGS.window_size)
    env = TemporalEnsembleWrapper(env, FLAGS.action_horizon)
    
    model = OctoModel.load_pretrained(
        FLAGS.checkpoint_weights_path,
        FLAGS.checkpoint_step,
    )

    def sample_actions(pretrained_model:OctoModel, observations, tasks, rng):
        observations = jax.tree_map(lambda x: x[None], observations)
        actions = pretrained_model.sample_actions(
            observations,
            tasks,
            rng=rng,
            unnormalization_statistics=pretrained_model.dataset_statistics["action"],
        )
        return actions[0]

    policy_fn = supply_rng(
        partial(
            sample_actions,
            model,
            # argmax=FLAGS.deterministic,
            # temperature=FLAGS.temperature,
            # argmax=False,
            # temperature=1.0,
        )
    )

    goal_image = jnp.zeros((FLAGS.im_size, FLAGS.im_size, 3), dtype=np.uint8)
    goal_instruction = ""

    while True:
        modality = click.prompt("Language or goal image?", type=click.Choice(["l", "g"]))
        
        if modality == "g":
            if click.confirm("Take a new goal?", default=True):
                assert isinstance(FLAGS.goal_pose, list)
                _pose = [float(e) for e in FLAGS.goal_pose]
                env.rob.movej(_pose, acc=FLAGS.acc, vel=FLAGS.vel)
                
                input("Press [Enter] when ready for taking the goal image.")
                obs = env.get_observation()
                goal = jax.tree_map(lambda x: x[None], obs)
            
            task = model.create_tasks(goals=goal)
            goal_image = goal["image_primary"][0]
            goal_instruction = ""

        elif modality == "l":
            print("Current instruction:", goal_instruction)
            if click.confirm("Take a new instruction?", default=True):
                text = input("Instruction?Please:")
            task = model.create_tasks(texts=[text])
            goal_instruction = text
            goal_image = jnp.zeros_like(goal_image)
        else:
            raise NotImplementedError()
        
        input("Press [Enter] to start.")
        
        obs, _ = env.reset()
        time.sleep(2.0)
        last_tstep = time.time()
        images = []
        goals = []
        t = 0
        while t < FLAGS.num_timesteps:
            if time.time() > last_tstep + STEP_DURATION:
                last_tstep = time.time()
                # print(obs["image_primary"].shape)
                images.append(obs["image_primary"][-1])
                goals.append(goal_image)

                if FLAGS.show_image:
                    bgr_img = cv2.cvtColor(obs["image_primary"][-1], cv2.COLOR_RGB2BGR)
                    cv2.imshow("img_view", obs["image_primary"][-1])
                    cv2.waitKey(20)

                forward_pass_time = time.time()
                action = np.array(policy_fn(obs, task), dtype=np.float64)
                print("forward pass time:", time.time() - forward_pass_time)

                start_time = time.time()
                obs, _, _, truncated, _ = env.step(action)
                print("step time:", time.time() - start_time)

                t += 1
                if truncated:
                    break
        if FLAGS.video_save_path is not None:
            os.makedirs(FLAGS.video_save_path, exist_ok=True)
            curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(
                FLAGS.video_save_path,
                f"{curr_time}.mp4",
            )
            video = np.concatenate([np.stack(goals), np.stack(images)], axis=1)
            imageio.mimsave(save_path, video, fps=1.0 / STEP_DURATION * 3)


if __name__ == "__main__":
    app.run(main)