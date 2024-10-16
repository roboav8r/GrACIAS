# situated_interaction
This is my development scratchpad - don't judge

# Setup

## System
This assumes that you have ROS2 Humble installed on Ubuntu 22.04 and have an NVidia GPU.

## Prerequisites
### Install Mamba (or another virtual environment manager of your choice)
Follow the instructions at https://github.com/conda-forge/miniforge:
```curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh # ...and follow the prompts
conda install mamba
```
### NVidia with Docker
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

## Installation
Create the workspace. At a terminal:
```
cd ~
mkdir -p sit_int_ws/src && cd ~/sit_int_ws/src
```

Clone the repo and its dependencies.
```
git clone https://github.com/ros-drivers/audio_common -b ros2
git clone https://github.com/roboav8r/situated_hri_interfaces -b devel
git clone https://github.com/roboav8r/tracking_msgs -b devel
git clone https://github.com/roboav8r/ar_track_alvar_msgs -b ros2
git clone https://github.com/roboav8r/marmot -b devel
git clone https://github.com/roboav8r/ros_audition -b devel
git clone https://github.com/roboav8r/mm_scene_rec -b devel
git clone https://github.com/roboav8r/2D_lidar_person_detection.git -b ros2
git clone https://github.com/roboav8r/situated_interaction

```
Create the virtual environment.
```
sudo apt-get install libasound2-dev ffmpeg
cd ~/sit_int_ws/src/situated_interaction
mamba env create -f sit_int_env.yml
mamba activate sit_int
conda remove --force ffmpeg
ffmpeg -devices # Optional - should have an "ALSA" device listed
```

```
mamba activate sit_int
source /opt/ros/humble/setup.bash
cd ~/sit_int_ws
rosdep install --from-paths src -y --ignore-src
colcon build --packages-select ar_track_alvar_msgs audio_common_msgs tracking_msgs # build messages and interfaces
source install/setup.bash
colcon build --packages-select situated_hri_interfaces
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

### Task 0b: Compute visual role recognition model (~12 hours)
Run the playback/model computation script
```
cd ~/sit_int_ws
mamba activate sit_int
source /opt/ros/humble/setup.bash
source install/setup.bash
cd ~/sit_int_ws/src/situated_interaction/scripts/training
python3 exp0b_compute_role_rec_models.py
```

### Task 0c: Evaluate speech recognition parameters (~4 minutes per configuration)
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

## Task 1: Evaluate audio, visual, and fused scene recognition (~17 hours)
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

## Task 2: Evaluate Bayesian role & command recognition

### 2a: Role recognition
Launch the Docker container
```
cd ~/sit_int_ws/src/situated_interaction/experiments/docker
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
ros2 launch situated_interaction exp2a_record_role_results.launch.py
```

### 2b: Command recognition
Launch the Docker container
```
cd ~/sit_int_ws/src/situated_interaction/experiments/docker
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
ros2 launch situated_interaction exp2b_record_cmd_results.launch.py
```

## Analyzing the data
```
mamba activate sit_int
cd ~/sit_int_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
jupyter-notebook
```

## Running the live demo
Launch the Docker container
```
cd ~/sit_int_ws/src/situated_interaction/docker/robot
docker compose up
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
