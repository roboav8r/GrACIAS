# situated_interaction
This is my development scratchpad - don't judge

# Setup

## Create environment
sudo apt-get install libasound2-dev
sudo apt-get install ffmpeg # 4.4.2, check alsa support with ffmpeg -devices
mamba env create -f sit_int_env.yml
mamba activate sit_int
conda remove --force ffmpeg

## NVidia with Docker
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

# Usage
## Task 0: Compute observation models

### Task 0a: Compute audio and visual scene recognition models
Run the ROS nodes
```
mamba activate sit_int
source /opt/ros/humble/setup.bash
cd ~/sit_int_ws
colcon build --packages-select ar_track_alvar_msgs audio_common_msgs situated_hri_interfaces tracking_msgs # build messages and interfaces
source install/setup.bash
colcon build --packages-select marmot mm_scene_rec ros_audition situated_interaction
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
jupyter-notebook
```


## Analysis / Experiment 1
Run the ROS nodes
```
mamba activate sit_int
source /opt/ros/humble/setup.bash
cd ~/sit_int_ws
colcon build --packages-select ar_track_alvar_msgs audio_common_msgs situated_hri_interfaces tracking_msgs # build messages and interfaces
source install/setup.bash
colcon build --packages-select marmot mm_scene_rec ros_audition situated_interaction
source install/setup.bash
ros2 launch situated_interaction analysis_1-1.launch.py
```
Run the analysis script
```
cd ~/sit_int_ws
mamba activate sit_int
source /opt/ros/humble/setup.bash
source install/setup.bash
jupyter-notebook
```

## Analysis - HRI data experiments
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

Replay the data
```
cd ~/sit_int_ws
mamba activate sit_int
source /opt/ros/humble/setup.bash 
source install/setup.bash
jupyter-notebook
```
- Navigate to `/src/situated_interaction/scripts/playback_and_recording`
- Run the `exp2_hri_playback.ipynb` notebook

# Future Work

## Debugging

## Improves/New features
- datatypes.SemanticObject: add continuous variables. Add id, auth, comms.
- For object/track message - instead of .image_available and .image, make generic feature vector dict with names, type
- Check weight division calcs
- Determine data type for mm scene rec, or make a param
- optimize torch -> np -> torch conversions in the scene recognition node
- Add permanent attribute option for - e.g. "movable" "animal" etc
- rename/rebrand node: semantic tracking? situated interaction node? hierarchical control node?

## Refactoring

## Demo ideas
- focus: attention at robot, away from robot as state
- team affiliation, e.g. longhorn or aggie