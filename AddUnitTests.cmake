# Function adding a suite of unit tests to the build
function(add_unit_tests)
    set(options)
    set(args LIB_NAME)
    set(list_args TEST_FILES)
    cmake_parse_arguments(
            PARSE_ARGV 0
            ADD_UNIT_TESTS
            "${options}"
            "${args}"
            "${list_args}"
    )

    foreach(arg IN LISTS ADD_UNIT_TESTS_UNPARSED_ARGUMENTS)
        message(WARNING "Unparsed argument: ${arg}")
    endforeach()

    # Add the unit tests executable and link the library being tested
    include_directories(${HOMING_PIGEON_ROOT}/libs/catch)
    include_directories(${HOMING_PIGEON_ROOT}/libs/hopi/tests)
    string(TOUPPER ${ADD_UNIT_TESTS_LIB_NAME} NAME)
    set(EXECUTABLE_NAME ${NAME}_UnitTests)
    add_executable (${EXECUTABLE_NAME} ${ADD_UNIT_TESTS_TEST_FILES})
    add_dependencies(${EXECUTABLE_NAME} ${ADD_UNIT_TESTS_LIB_NAME})
    target_link_libraries(${EXECUTABLE_NAME} PUBLIC ${ADD_UNIT_TESTS_LIB_NAME})
endfunction()
