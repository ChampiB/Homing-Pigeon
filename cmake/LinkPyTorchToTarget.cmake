# Function linking pytorch to the target whose name is passed as parameters
function(link_pytorch_to_target)
    set(args TARGET TORCH_DIR)
    cmake_parse_arguments(PARSE_ARGV 0 "ARG" "" "${args}" "")

    foreach(arg IN LISTS ARG_UNPARSED_ARGUMENTS)
        message(WARNING "Unparsed argument: ${arg}")
    endforeach()

    # Link boost
    find_package(Boost 1.40.0 COMPONENTS filesystem system iostreams REQUIRED)
    target_link_libraries(${ARG_TARGET} PUBLIC "${Boost_LIBRARIES}")

    # Link pytorch
    set(Torch_DIR  "${ARG_TORCH_DIR}/share/cmake/Torch")
    set(Caffe2_DIR "${ARG_TORCH_DIR}/share/cmake/Caffe2")
    find_package(Torch REQUIRED)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    target_link_libraries(${ARG_TARGET} PUBLIC "${TORCH_LIBRARIES}")

    if (MSVC)
        file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
        add_custom_command(TARGET ${ARG_TARGET}
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TORCH_DLLS}
                $<TARGET_FILE_DIR:${ARG_TARGET}>)
    endif (MSVC)

endfunction()
