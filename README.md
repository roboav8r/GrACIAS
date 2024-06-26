# situated_interaction
This is my development scratchpad - don't judge

# Setup

## Create environment
sudo apt-get install libasound2-dev
sudo apt-get install ffmpeg # 4.4.2, check alsa support with ffmpeg -devices
mamba env create -f sit_int_env.yml
mamba activate sit_int
conda remove --force ffmpeg


# Future Work


## Debugging
- check update time/image

## Improves/New features
- Add multiple factors to attributes instead of continuous recursive bayesian update
- allow attributes or states to be updated by confidence OR by timeout
- Check weight division calcs
- Determine data type for mm scene rec, or make a param
- optimize torch -> np -> torch conversions in the scene recognition node
- Add permanent attribute option for - e.g. "movable" "animal" etc
- cap confidence at 99 to prevent numerical instability in mm_scene_rec and situated_interaction
- rename/rebrand node: semantic tracking? situated interaction node? hierarchical control node?

## Refactoring

## Demo ideas
- focus: attention at robot, away from robot as state
- team affiliation, e.g. longhorn or aggie