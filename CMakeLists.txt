cmake_minimum_required(VERSION 3.17)

project(HomingPigeon VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
SET(HOMING_PIGEON_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

add_definitions(-DNOMINMAX)

# Includes
include(AddExample.cmake)
include(AddUnitTests.cmake)

# Library: Homing Pigeon
add_subdirectory(libs/hopi)

# Examples of Homing Pigeon
add_example(NAME maze_environment)
add_example(NAME maze_navigation)
add_example(NAME learning_maze_navigation)
add_example(NAME factor_graph_visualisation)
add_example(NAME tensorflow_main)
