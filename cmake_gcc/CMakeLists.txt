# Define minimal required version of CMake.
cmake_minimum_required(VERSION "3.25")

# Project definition
project(
	aiml_soc_magic_wand_baremetal
	VERSION 1.0
	LANGUAGES C CXX ASM
)

# Include the definition of the slc_aiml_soc_magic_wand_baremetal target,
# which contains the content of the SLC project
include(aiml_soc_magic_wand_baremetal.cmake)

get_target_property(_slc_sources slc SOURCES)
set(_si7210_sdk_source "")
foreach(_source IN LISTS _slc_sources)
    if(_source MATCHES "/board_drivers/hardware/driver/si7210/src/sl_si7210\\.c$")
        set(_si7210_sdk_source "${_source}")
        break()
    endif()
endforeach()

if(NOT _si7210_sdk_source)
    message(FATAL_ERROR "Could not find the SDK Si7210 driver source to replace")
endif()

list(REMOVE_ITEM _slc_sources "${_si7210_sdk_source}")
set_property(TARGET slc PROPERTY SOURCES "${_slc_sources}")
target_sources(slc PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../components/si7210/src/sl_si7210.c"
)

add_executable(aiml_soc_magic_wand_baremetal
    # Add additional sources here
    ../fedppe_mcu.c
)

target_include_directories(aiml_soc_magic_wand_baremetal PUBLIC
    # Add additional include paths here
    ..
)

target_compile_definitions(aiml_soc_magic_wand_baremetal PUBLIC
    # Add additional macros here
)

target_compile_options(aiml_soc_magic_wand_baremetal PUBLIC
    # Set additional compiler flags here
)

target_link_options(aiml_soc_magic_wand_baremetal PUBLIC
    # Set additional linker flags here
)

# Link with the content defined in the SLC project
target_link_libraries(aiml_soc_magic_wand_baremetal PRIVATE
    slc
)

# Include managed project content if available
include(aiml_soc_magic_wand_baremetal_project.cmake OPTIONAL RESULT_VARIABLE managed_project)
if(managed_project)
    message(STATUS "Using managed project content from ${managed_project}")
endif()

# Force the gcc linker command
set_target_properties(aiml_soc_magic_wand_baremetal PROPERTIES LINKER_LANGUAGE C)
set_target_properties(aiml_soc_magic_wand_baremetal PROPERTIES OUTPUT_NAME "tcc_001")

# Create .bin, .hex and .s37 artifacts after building the project
add_custom_command(TARGET aiml_soc_magic_wand_baremetal
    POST_BUILD
    COMMAND ${CMAKE_OBJCOPY} ${OBJCOPY_SREC_CMD} "$<TARGET_FILE:aiml_soc_magic_wand_baremetal>" "$<TARGET_FILE_DIR:aiml_soc_magic_wand_baremetal>/$<TARGET_FILE_BASE_NAME:aiml_soc_magic_wand_baremetal>.s37"
    COMMAND ${CMAKE_OBJCOPY} ${OBJCOPY_IHEX_CMD} "$<TARGET_FILE:aiml_soc_magic_wand_baremetal>" "$<TARGET_FILE_DIR:aiml_soc_magic_wand_baremetal>/$<TARGET_FILE_BASE_NAME:aiml_soc_magic_wand_baremetal>.hex"
    COMMAND ${CMAKE_OBJCOPY} ${OBJCOPY_BIN_CMD}  "$<TARGET_FILE:aiml_soc_magic_wand_baremetal>" "$<TARGET_FILE_DIR:aiml_soc_magic_wand_baremetal>/$<TARGET_FILE_BASE_NAME:aiml_soc_magic_wand_baremetal>.bin" 
)

# Run post-build pipeline to perform additional post-processing
if(post_build_command)
add_custom_command(TARGET aiml_soc_magic_wand_baremetal
    POST_BUILD
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
    COMMAND ${post_build_command}
)
endif()