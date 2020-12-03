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

    # Add the executable and link the hopi library
    add_executable(${ADD_EXAMPLE_NAME} ${HOMING_PIGEON_ROOT}/examples/${ADD_EXAMPLE_NAME}.cpp)
    add_dependencies(${ADD_EXAMPLE_NAME} hopi)
    target_link_libraries(${ADD_EXAMPLE_NAME} PUBLIC hopi)
    target_include_directories(${ADD_EXAMPLE_NAME} PUBLIC ${HOMING_PIGEON_ROOT}/libs/eigen)
endfunction()
