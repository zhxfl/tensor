project(Tests)
cmake_minimum_required(VERSION 2.8)
include_directories("${PROJECT_SOURCE_DIR}/common/*.hpp")
include_directories("${PROJECT_SOURCE_DIR}/tensor/*.hpp")
include_directories("${PROJECT_SOURCE_DIR}/gtest/*.h")
Message("${PROJECT_BINARY_DIR}/gtest/")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")

link_directories("${PROJECT_BINARY_DIR}/gtest")
# Function prepares name of a test executable
#    @output_name -  output variable's name
#    @filename    -  test_*.cpp file path
function(test_name output_name filename)
    get_filename_component(name ${filename} NAME_WE)
    set(${output_name} ${name}${TEST_EXT} PARENT_SCOPE)
endfunction()

file(GLOB TEST_CPP_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "test_*.cpp")

foreach(source ${TEST_CPP_SOURCES})
  MESSAGE( STATUS ${source} )
endforeach()

#    Build each test separately from *.cpp files
foreach(source ${TEST_CPP_SOURCES})
    test_name(TEST_NAME ${source})
    add_executable(${TEST_NAME} ${source})
    target_link_libraries(${TEST_NAME} 
        tensor
        gtest
        pthread
        )
endforeach()

