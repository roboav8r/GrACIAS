cmake_minimum_required(VERSION 3.8)
project(GrACIAS)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Install launch and config files
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(tracking_msgs REQUIRED)

# Install C++ execs
add_executable(track_preproc src/track_preproc.cpp)
ament_target_dependencies(track_preproc rclcpp tracking_msgs)
install(TARGETS 
  track_preproc
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python
# ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS
  scripts/int_mgr_node.py
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
