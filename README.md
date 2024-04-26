# situated_interaction

# Setup

## Create environment
sudo apt-get install libasound2-dev
sudo apt-get install ffmpeg # 4.4.2, check alsa support with ffmpeg -devices
mamba env create -f sit_int_env.yml


# Improves
- Check weight division calcs
- Determine data type, or make a param
- optimize torch -> np -> torch conversions in the scene recognition node