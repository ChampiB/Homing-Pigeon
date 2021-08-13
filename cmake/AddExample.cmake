# Function adding an example to the build
function(add_example)
    set(args NAME HOPI_PROJECT_DIR)
    cmake_parse_arguments(
            PARSE_ARGV 0
            ARG
            ""
            "${args}"
            ""
    )

    foreach(arg IN LISTS ARG_UNPARSED_ARGUMENTS)
        message(WARNING "Unparsed argument: ${arg}")
    endforeach()

    # Add the executable
    add_executable(${ARG_NAME} ${ARG_HOPI_PROJECT_DIR}/examples/${ARG_NAME}.cpp)

    # Link hopi to executable
    target_link_libraries(${ARG_NAME} PRIVATE hopi)
endfunction()
