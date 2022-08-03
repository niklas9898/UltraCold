#######################################################
# Configure and build cuda libraries for Ultracold
#######################################################

#--------------
# Enable CUDA
#--------------

find_package(CUDA 11.0 REQUIRED)
enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)

#-----------------------
# Create the library
#-----------------------

add_library(cudaSolvers
        STATIC
        ${CMAKE_SOURCE_DIR}/source/cuda/cudaDeviceVector.cu
        ${CMAKE_SOURCE_DIR}/source/cuda/cudaGPSolver.cu
)

#----------------------------
# Set the include directories
#----------------------------

target_include_directories(cudaSolvers
        INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

#---------------------------
# Link internal libraries
#---------------------------

target_link_libraries(cudaSolvers PUBLIC utilities)

#----------------------
# Link CUDA libraries
#----------------------
target_link_libraries(cudaSolvers PUBLIC ${CUDA_LIBRARIES})

#---------------------------------------
# Allow CUDA to resolve its own symbols
#---------------------------------------

set_target_properties(cudaSolvers
        PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)

#-----------------------------------------------------------------
# Suppress generation of specific code for specific architecture
#-----------------------------------------------------------------

set_property(TARGET cudaSolvers
        PROPERTY CUDA_ARCHITECTURES
        OFF
)

#---------------------------------------
# Enable separable compilation
#---------------------------------------

set_property(TARGET cudaSolvers
        PROPERTY
        CUDA_SEPARABLE_COMPILATION ON
)

#----------------------------------------------------
# Install libraries and files to the final location
#----------------------------------------------------

install(TARGETS cudaSolvers
        DESTINATION lib
        EXPORT UltraColdTargets
)

install(FILES cudaDeviceVector.cuh DESTINATION include)