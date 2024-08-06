# situated_interaction
This is my development scratchpad - don't judge

# Setup

## Create environment
sudo apt-get install libasound2-dev
sudo apt-get install ffmpeg # 4.4.2, check alsa support with ffmpeg -devices
mamba env create -f sit_int_env.yml
mamba activate sit_int
conda remove --force ffmpeg

# Usage

## Analysis / Experiment 1
```
mamba activate sit_int
source /opt/ros/humble/setup.bash
cd ~/sit_int_ws
colcon build --packages-select ar_track_alvar_msgs audio_common_msgs situated_hri_interfaces tracking_msgs # build messages and interfaces
source install/setup.bash
colcon build --packages-select marmot mm_scene_rec ros_audition situated_interaction
source install/setup.bash
ros2 launch situated_interaction analysis.launch.py
```

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