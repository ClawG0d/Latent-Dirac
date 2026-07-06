cmake_minimum_required(VERSION 3.16)
project(transformer CXX)

# First-party engine-track application; links against an installed
# vanilla Geant4 (point CMAKE_PREFIX_PATH or Geant4_DIR at the install).
find_package(Geant4 REQUIRED)

add_executable(transformer transformer.cc)
target_link_libraries(transformer PRIVATE ${Geant4_LIBRARIES})
target_include_directories(transformer PRIVATE ${Geant4_INCLUDE_DIRS})
set_target_properties(transformer PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
