cmake_minimum_required(VERSION 3.15)
project(snn_core_c_embedded C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Path to snn_core headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include)

# Detection of static library compiled by Cargo
set(SNN_CORE_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../target/release)
find_library(SNN_CORE_LIB
    NAMES snn_core
    PATHS ${SNN_CORE_LIB_DIR}
    NO_DEFAULT_PATH
)

add_executable(c_embedded_demo main.c)

if(SNN_CORE_LIB)
    target_link_libraries(c_embedded_demo PRIVATE ${SNN_CORE_LIB} -lpthread -ldl -lm)
    message(STATUS "Found libsnn_core: ${SNN_CORE_LIB}")
else()
    message(WARNING "libsnn_core.a not found in target/release. Run 'cargo rustc --release --crate-type staticlib' first.")
endif()
