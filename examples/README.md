<<<<<<< HEAD
## Examples

We provide simple [example scripts](examples) that demonstrate how to inference and finetune OCTO models,
as well as how to use our data loader independently. We provide the following examples:

|                                                                      |                                                                                                                 |
|----------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| [OCTO Inference](examples/01_inference_pretrained.ipynb)             | Minimal example for loading and inferencing a pre-trained OCTO model                                            |
| [OCTO Finetuning](examples/02_finetune_new_observation_action.py)    | Minimal example for finetuning a pre-trained OCTO models on a small dataset with new observation + action space |
| [OCTO Rollout](examples/03_eval_finetuned.py)                        | Run a rollout of a pre-trained OCTO policy in a Gym environment                                                 |
| [OCTO Robot Eval](examples/04_eval_finetuned_on_robot.py)            | Evaluate a pre-trained OCTO model on a real WidowX robot                                                        |
| [OpenX Dataloader Intro](examples/05_dataloading.ipynb)              | Walkthrough of the features of our Open X-Embodiment data loader                                                |
| [OpenX PyTorch Dataloader](examples/06_pytorch_oxe_dataloader.ipynb) | Standalone Open X-Embodiment data loader in PyTorch                                                             |
=======
| Example                              | Description                                                  | Notes                                                        |
| ------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| align_filter_viewer.py               | Demonstrates how to use the align filter.                    | Supported by the Gemini 330 series.                          |
| color_viewer.py                      | Displays the color stream from the camera.                   |                                                              |
| depth_color_sync.py                  | Demonstrates how to synchronize the depth and color streams. |                                                              |
| depth_viewer.py                      | Displays the depth stream from the camera.                   |                                                              |
| depth_viewer_callback.py             | Displays the depth stream from the camera using a callback.  |                                                              |
| depth_work_mode.py                   | Demonstrates how to set the depth work mode.                 |                                                              |
| double_infrared_viewer.py            | Demonstrates how to display the double infrared stream.      | Supported by the Gemini 2 series and  Gemini 330 series.     |
| hdr_merge_filter.py                  | Demonstrates how to merge HDR images.                        | Supported by the Gemini 330 series.                          |
| hello_orbbec.py                      | Demonstrates how to obtain device information.               |                                                              |
| hot_plug.py                          | Demonstrates how to detect hot plug events.                  |                                                              |
| imu_reader.py                        | Demonstrates how to read IMU data.                           |                                                              |
| infrared_viewer.py                   | Displays the infrared stream from the camera.                | Not supported by the  Gemini 2 series and  Gemini 330 serie. Refer to double_infrared_viewer.py for alternatives. |
| multi_device.py                      | Demonstrates how to use multiple devices.                    |                                                              |
| net_device.py                        | Demonstrates how to use network functions.                   | Supported by Femto Mega and Gemini 2 XL.                     |
| playback.py                          | Demonstrates how to play back recorded streams.              |                                                              |
| pointcloud_filter_o3d.py             | Demonstrates how to display the point cloud.                 | Supported by the Gemini 330 series. Requires installation of Open3D. |
| post_process.py                      | Demonstrates how to use post-processing filters.             | Supported by the Gemini 330 series.                          |
| recorder.py                          | Demonstrates how to record the depth and color streams to a file. |                                                              |
| save_color_points_to_disk.py         | Demonstrates how to save the depth and color streams to disk. |                                                              |
| save_pointcloud_to_disk.py           | Demonstrates how to save the point cloud to disk; the data is a numpy array. | Not supported by the Gemini 330 series.                      |
| save_pointcloud_to_disk_by_filter.py | Demonstrates how to save the point cloud to disk using a point cloud filter. | Supported by the Gemini 330 series.                          |
| set_data.py                          | Demonstrates how to set data.                                | Not supported by the Gemini 330 series.                      |
| set_depth_unit.py                    | Demonstrates how to set the depth unit.                      | Supported by the Gemini 2 series and Gemini 330 series.      |
| two_devices_sync.py                  | Demonstrates how to synchronize two devices.                 |                                                              |

>>>>>>> pyorbbecsdk-remote/main
