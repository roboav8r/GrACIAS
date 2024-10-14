FROM ros:galactic-ros-base-focal 
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# setup ros1 sources.list and keys 
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list 
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 

ENV ROS1_DISTRO noetic 
ENV ROS2_DISTRO galactic 

# install ros1 debian packages 
RUN apt-get update && apt-get install -y --no-install-recommends \
wget pip \
ros-noetic-ros-comm \
ros-noetic-tf2-msgs \
ros-noetic-cmake-modules \ 
ros-noetic-cv-bridge \
ros-noetic-dynamic-reconfigure \
ros-noetic-image-transport \
ros-noetic-pcl-conversions \
ros-noetic-pcl-ros \
ros-noetic-resource-retriever \
ros-noetic-visualization-msgs \
ros-noetic-pointcloud-to-laserscan \
ros-noetic-roscpp-tutorials=0.10.2-1* \
ros-noetic-rospy-tutorials=0.10.2-1* \
&& rm -rf /var/lib/apt/lists/* 

# Clone and build custom ROS1 messages from source 
RUN mkdir -p /ros1_ws/src 
WORKDIR /ros1_ws/src 
RUN git clone -b noetic-devel https://github.com/ros-perception/ar_track_alvar.git
RUN git clone https://github.com/roboav8r/2D_lidar_person_detection.git
WORKDIR /ros1_ws/src/2D_lidar_person_detection/dr_spaam
RUN python3 setup.py install
WORKDIR /ros1_ws 
# RUN wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1gH_pdxUC8diIFUe3_Ky8QiVVCyiR3500' -O ckpt_jrdb_ann_dr_spaam_e20.pth
# RUN wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1QuB9mwm8h46vIiynFsDKKxlEiUYC0quN' -O ckpt_jrdb_ann_drow3_e40.pth
# RUN wget https://drive.google.com/drive/folders/1OI99VfUBkmRSijgmMYYku9Pc_nS3v8sj/ckpt_jrdb_ann_dr_spaam_e20.pth
# RUN wget https://drive.google.com/drive/folders/1OI99VfUBkmRSijgmMYYku9Pc_nS3v8sj/ckpt_jrdb_ann_drow3_e40.pth
RUN source /opt/ros/${ROS1_DISTRO}/setup.bash && \ 
catkin_make

# dr-spaam detector requirements
RUN pip install networkx==2.8.8 
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip install scipy

COPY ./ros1_ar_entrypoint.sh /
COPY ./ros1_pcl_to_scan_entrypoint.sh /
COPY ./ros1_leg_detector_entrypoint.sh /
COPY ./ros1_master_entrypoint.sh /