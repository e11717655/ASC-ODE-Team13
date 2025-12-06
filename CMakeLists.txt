cmake_minimum_required(VERSION 3.20)
project(ASC_ode)

set (CMAKE_CXX_STANDARD 20)


include_directories(src nanoblas/src)

find_package(Eigen3 REQUIRED)
include_directories(src nanoblas/src ${EIGEN3_INCLUDE_DIRS})

add_subdirectory (src)
add_subdirectory (nanoblas)
add_subdirectory (mechsystem)


add_executable(mass_spring_demo mechsystem/mass_spring.cpp)
target_link_libraries(mass_spring_demo PUBLIC nanoblas Eigen3::Eigen)

add_executable(legendre demos/legendre.cpp)
target_link_libraries(legendre PUBLIC nanoblas)

add_executable (test_ode demos/test_ode.cpp)
target_link_libraries (test_ode PUBLIC nanoblas)

add_executable (demo_autodiff demos/demo_autodiff.cpp)
target_link_libraries (demo_autodiff PUBLIC nanoblas)

add_executable (test_autodiff demos/test_autodiff.cpp)
target_link_libraries (test_autodiff PUBLIC nanoblas)

