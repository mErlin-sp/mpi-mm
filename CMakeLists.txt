cmake_minimum_required(VERSION 3.22)
project(mpi_mm C)

set(CMAKE_C_STANDARD 23)

# Set the CMAKE_C_FLAGS variable
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,--stack,100000000")

# Set source files
set(SOURCES mpi_mm_blocking.c)

# Create executable target
add_executable(${PROJECT_NAME} ${SOURCES})

# Find MPI_hwloc_LIBRARY package
#find_library(MPI_hwloc_LIBRARY NAMES cyghwloc-15 PATHS /usr/bin/)

# Find MPI package
find_package(MPI REQUIRED)

# Link with MPI library
target_link_libraries(${PROJECT_NAME} MPI::MPI_C)
#target_link_libraries(${PROJECT_NAME} ${MPI_hwloc_LIBRARY})




