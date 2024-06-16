cmake_minimum_required(VERSION 3.8)
project(situated_interaction)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Install launch and config files
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY pretrained_models DESTINATION share/${PROJECT_NAME}/)
# install(DIRECTORY datasets DESTINATION share/${PROJECT_NAME}/)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(tracking_msgs REQUIRED)
find_package(ar_track_alvar_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(foxglove_msgs)
find_package(situated_hri_interfaces)
find_package(geometry_msgs)
find_package(tf2_ros)
find_package(tf2_geometry_msgs)

# Install C++ execs
add_executable(track_preproc src/track_preproc.cpp)
ament_target_dependencies(track_preproc rclcpp tracking_msgs)
add_executable(ar_preproc src/artag_preproc.cpp)
ament_target_dependencies(ar_preproc tf2_ros tf2_geometry_msgs rclcpp ar_track_alvar_msgs visualization_msgs situated_hri_interfaces foxglove_msgs geometry_msgs)
install(TARGETS 
  track_preproc
  ar_preproc
  DESTINATION lib/${PROJECT_NAME}
)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/PersonVisRec.srv"
  DEPENDENCIES builtin_interfaces sensor_msgs # Add packages that above messages depend on
)


# Install Python
ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS
  scripts/int_mgr_node.py
  scripts/person_vis_rec_server.py
  # scripts/audio_acq_node.py
  scripts/audio_bf_node.py
  # scripts/audio_proc_node.py
  scripts/scene_rec_node.py
  scripts/scene_voice_node.py
  DESTINATION lib/${PROJECT_NAME}
)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
