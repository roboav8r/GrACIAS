FROM ros:humble-ros-base-jammy
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# install ros2 packages
# TODO - replace with rosdep RUN rosdep install -i --from-path src --rosdistro humble -y
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-foxglove-msgs \
    ros-humble-vision-msgs \
    ros-humble-foxglove-bridge \
    wget \ 
    # ros-humble-desktop=0.10.0-1* \
    && rm -rf /var/lib/apt/lists/*

# Build packages from source
RUN mkdir -p /perception_ws/src 
WORKDIR /perception_ws/src 

# Clone packages
RUN git clone -b ros2 https://github.com/roboav8r/tracking_msgs.git
RUN git clone https://github.com/roboav8r/marmot.git
RUN git clone -b hri-demo-devel https://github.com/roboav8r/situated_interaction.git
RUN git clone https://github.com/roboav8r/ar_track_alvar_msgs.git
RUN git clone https://github.com/roboav8r/gracias_interfaces.git
WORKDIR /perception_ws 

# Setup conda environment
RUN mkdir -p /opt/conda 
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_4.12.0-Linux-x86_64.sh -O /opt/conda/miniconda.sh && bash /opt/conda/miniconda.sh -b -p /opt/miniconda 
# # Uncomment once you are ready to start productionizing the image 
# # COPY /perception_ws/src/situated_interaction/gracias_env.yaml /tmp 
RUN . /opt/miniconda/bin/activate && conda env update --name base --file /perception_ws/src/situated_interaction/hri_env.yml

# Build packages
RUN source /opt/ros/humble/setup.bash && colcon build --packages-select tracking_msgs gracias_interfaces ar_track_alvar_msgs
RUN source install/setup.bash && colcon build


COPY ./perception_entrypoint.sh /