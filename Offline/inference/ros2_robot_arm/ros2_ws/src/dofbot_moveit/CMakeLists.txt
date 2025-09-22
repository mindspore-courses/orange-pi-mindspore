cmake_minimum_required(VERSION 3.8)
project(dofbot_moveit)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(dofbot_info REQUIRED)

set(dependencies 
  rclcpp
  kdl_parser
  dofbot_info
)

include_directories(
  include/${PROJECT_NAME}
)

add_executable(dofbot_server src/dofbot_server.cpp)
target_link_libraries(dofbot_server dofbot_kinemarics)
ament_target_dependencies(dofbot_server ${dependencies})

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

install(TARGETS ${library_name}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

install(TARGETS dofbot_server
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(${library_name})
ament_export_dependencies(${dependencies})

ament_package()

install(DIRECTORY launch
DESTINATION share/${PROJECT_NAME})

install(DIRECTORY rviz
DESTINATION share/${PROJECT_NAME})

install(DIRECTORY urdf
DESTINATION share/${PROJECT_NAME})

install(DIRECTORY meshes
DESTINATION share/${PROJECT_NAME})
