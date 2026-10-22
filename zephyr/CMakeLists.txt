# CMakeLists.txt - Zephyr RTOS integration for snn_core

if(CONFIG_SNN_CORE)
    zephyr_interface_library_named(snn_core)

    target_include_directories(snn_core INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )

    zephyr_library()
    zephyr_library_link_libraries(snn_core)

    message(STATUS "snn_core: Neuromorphic inference engine enabled in Zephyr RTOS")
endif()
