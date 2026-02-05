cmake_minimum_required(VERSION 3.16)

project(GrupalMPI)

find_package(fmt CONFIG REQUIRED)

#  -- conf MPI
set(MPI_ROOT "C:/oneAPI/mpi/latest")
set(MPI_INCLUDE "${MPI_ROOT}/include")
# Libreria para enlazar
set(MPI_LIB "${MPI_ROOT}/lib")

include_directories(${MPI_INCLUDE}) # Aqui incluye archivos .h

add_executable(grupalMPI 
main.cpp
blelloch.cpp)

target_link_libraries(grupalMPI PRIVATE 
fmt::fmt 
${MPI_LIB}/impi.lib)
