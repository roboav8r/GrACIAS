# situated_interaction
This is my development scratchpad - don't judge

# Setup

## Prerequisites
This assumes that you have ROS2 Humble installed on Ubuntu 22.04 and have an NVidia GPU.

## Create environment
sudo apt-get install libasound2-dev
sudo apt-get install ffmpeg
mamba env create -f sit_int_env.yml
mamba activate sit_int
conda remove --force ffmpeg
ffmpeg -devices # Optional - should have an "ALSA" device listed

## NVidia with Docker
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

## Clone and build the repo
```
mamba activate sit_int
source /opt/ros/humble/setup.bash
cd ~/sit_int_ws
colcon build --packages-select ar_track_alvar_msgs audio_common_msgs situated_hri_interfaces tracking_msgs # build messages and interfaces
source install/setup.bash
colcon build --packages-select marmot mm_scene_rec ros_audition situated_interaction
source install/setup.bash
```

# Usage
## Task 0: Compute observation models

### Task 0a: Compute audio and visual scene recognition models (~16 hours)
Run the ROS nodes
```
cd ~/sit_int_ws
mamba activate sit_int
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch situated_interaction exp0a_scene_model_nodes.launch.py
```

In a separate window, run the playback/model computation script
```
cd ~/sit_int_ws
mamba activate sit_int
source /opt/ros/humble/setup.bash
source install/setup.bash
cd ~/sit_int_ws/src/situated_interaction/scripts/training
python3 exp0a_compute_scene_rec_models.py
```

### Task 0b: Compute visual role recognition model
Run the playback/model computation script
```
cd ~/sit_int_ws
mamba activate sit_int
source /opt/ros/humble/setup.bash
source install/setup.bash
cd ~/sit_int_ws/src/situated_interaction/scripts/training
python3 exp0b_compute_role_rec_models.py
```

### Task 0c: Evaluate speech recognition parameters
Run the playback/model computation script
```
cd ~/sit_int_ws
mamba activate sit_int
source /opt/ros/humble/setup.bash
source install/setup.bash
cd ~/sit_int_ws/src/situated_interaction/scripts/training
python3 exp0c_evaluate_speech_rec.py
```

## Evaluate Perception System

## Task 1: Evaluate audio, visual, and fused scene recognition - TODO
Run the ROS nodes
```
mamba activate sit_int
source /opt/ros/humble/setup.bash
cd ~/sit_int_ws
colcon build --packages-select ar_track_alvar_msgs audio_common_msgs situated_hri_interfaces tracking_msgs # build messages and interfaces
source install/setup.bash
colcon build --packages-select marmot mm_scene_rec ros_audition situated_interaction
source install/setup.bash
ros2 launch situated_interaction exp1_record_scene_results.launch.py
```
Run the analysis script
```
cd ~/sit_int_ws
mamba activate sit_int
source /opt/ros/humble/setup.bash
source install/setup.bash
jupyter-notebook
```

## Task 2: Evaluate simultaneous role & command recognition
Launch the Docker container
```
cd ~/sit_int_ws/src/situated_interaction/docker
docker compose up
```

Run the ROS nodes
```
mamba activate sit_int
source /opt/ros/humble/setup.bash
cd ~/sit_int_ws
colcon build --packages-select ar_track_alvar_msgs audio_common_msgs situated_hri_interfaces tracking_msgs # build messages and interfaces
source install/setup.bash
colcon build --packages-select marmot mm_scene_rec ros_audition situated_interaction
source install/setup.bash
ros2 launch situated_interaction exp2_record_hri_results.launch.py
```

## Analyzing the data
```
mamba activate sit_int
cd ~/sit_int_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
jupyter-notebook
```


# Future Work

## Debugging

## Improves/New features
- python3 exp0a_compute_scene_rec_models.py to exp manager node

- datatypes.SemanticObject: add continuous variables. Add id, auth, comms.
- For object/track message - instead of .image_available and .image, make generic feature vector dict with names, type
- Check weight division calcs
- Determine data type for mm scene rec, or make a param
- optimize torch -> np -> torch conversions in the scene recognition node
- Add permanent attribute option for - e.g. "movable" "animal" etc
- rename/rebrand node: semantic tracking? situated interaction node? hierarchical control node?

## Refactoring
- set variable for detections
- function: update_states(detections, assignments)

## Demo ideas
- focus: attention at robot, away from robot as state
- team affiliation, e.g. longhorn or aggie