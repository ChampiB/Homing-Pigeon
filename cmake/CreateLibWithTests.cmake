# Function adding a library to be build and the associated unit tests
function(create_lib_with_tests)
    # First level of arguments
    set(args CATCH_DIR)
    set(groups LIB TESTS)
    cmake_parse_arguments(GRP "" "${args}" "${groups}" "${ARGN}")

    # Second level of arguments
    set(args TARGET)
    set(list_args SOURCES PRIVATE_LIBS PUBLIC_LIBS PUBLIC_INCLUDE_DIRS)
    cmake_parse_arguments(LIB  "" "${args}" "${list_args}" "${GRP_LIB}")
    cmake_parse_arguments(TEST "" "${args}" "${list_args}" "${GRP_TESTS}")

    # Create the library
    add_library(${LIB_TARGET} ${LIB_SOURCES})
    target_link_libraries(${LIB_TARGET}
            PUBLIC ${LIB_PUBLIC_LIBS}
            PRIVATE ${LIB_PRIVATE_LIBS}
    )
    target_include_directories(${LIB_TARGET} PUBLIC ${LIB_PUBLIC_INCLUDE_DIRS})

    # Create the tests of the library
    if (NOT DEFINED TEST_TARGET)
        message(STATUS "No tests for the library: ${LIB_TARGET}.")
        return()
    endif()
    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_link_libraries(${TEST_TARGET}
            PUBLIC ${TEST_PUBLIC_LIBS}
            PRIVATE ${LIB_TARGET} ${TEST_PRIVATE_LIBS}
    )
    target_include_directories(${TEST_TARGET} PUBLIC ${TEST_PUBLIC_INCLUDE_DIRS})
    target_include_directories(${TEST_TARGET} PRIVATE ${GRP_CATCH_DIR})
endfunction()
