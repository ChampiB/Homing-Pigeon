# Function adding an example to the build
function(add_experiment)
    set(args NAME)
    set(list_args)
    cmake_parse_arguments(
            PARSE_ARGV 0
            ARG
            ""
            "${args}"
            "${list_args}"
    )

    foreach(arg IN LISTS ARG_UNPARSED_ARGUMENTS)
        message(WARNING "Unparsed argument: ${arg}")
    endforeach()

    # Add the executable
    add_executable(${ARG_NAME} experiments/${ARG_NAME}.cpp)

    # Link experiments library
    target_link_libraries(${ARG_NAME} PRIVATE experiments)
endfunction()
