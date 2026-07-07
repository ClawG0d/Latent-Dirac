cmake_minimum_required(VERSION 3.16)
project(positrongen CXX)

# First-party engine-track application; links against an installed
# vanilla Geant4 (point CMAKE_PREFIX_PATH or Geant4_DIR at the install).
find_package(Geant4 REQUIRED)

add_executable(positrongen positrongen.cc)
target_link_libraries(positrongen PRIVATE ${Geant4_LIBRARIES})
target_include_directories(positrongen PRIVATE ${Geant4_INCLUDE_DIRS})
set_target_properties(positrongen PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
