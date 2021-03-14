# Function adding an example to the build
function(add_example)
    set(options)
    set(args NAME)
    set(list_args)
    cmake_parse_arguments(
            PARSE_ARGV 0
            ADD_EXAMPLE
            "${options}"
            "${args}"
            "${list_args}"
    )

    foreach(arg IN LISTS ADD_EXAMPLE_UNPARSED_ARGUMENTS)
        message(WARNING "Unparsed argument: ${arg}")
    endforeach()

    # Add the executable
    add_executable(${ADD_EXAMPLE_NAME} ${HOMING_PIGEON_ROOT}/examples/${ADD_EXAMPLE_NAME}.cpp)

    # Link hopi
    add_dependencies(${ADD_EXAMPLE_NAME} hopi)
    target_link_libraries(${ADD_EXAMPLE_NAME} PUBLIC hopi)

    # Link pytorch
    set(Torch_DIR "${HOMING_PIGEON_ROOT}/libs/torch/share/cmake/Torch")
    set(Caffe2_DIR "${HOMING_PIGEON_ROOT}/libs/torch/share/cmake/Caffe2")

    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    target_link_libraries(hopi PUBLIC "${TORCH_LIBRARIES}")

    if (MSVC)
        file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
        add_custom_command(TARGET hopi
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TORCH_DLLS}
                $<TARGET_FILE_DIR:hopi>)
    endif (MSVC)

endfunction()
