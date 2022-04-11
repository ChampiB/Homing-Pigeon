cmake_minimum_required(VERSION 3.16)

project(HomingPigeon VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG -g)
add_definitions(-DNOMINMAX)

# Useful variables
set(HOPI_PROJECT_ROOT  ${CMAKE_CURRENT_SOURCE_DIR})
set(LIB_HOPI_ROOT      ${HOPI_PROJECT_ROOT}/libs/hopi)
set(LIB_CATCH_ROOT     ${HOPI_PROJECT_ROOT}/libs/catch)
set(LIB_TORCH_ROOT     ${HOPI_PROJECT_ROOT}/libs/torch)
set(HOPI_SRCS_PATH  "srcs/")
set(HOPI_TESTS_PATH "tests/")

# Includes
list(APPEND CMAKE_MODULE_PATH "${HOPI_PROJECT_ROOT}/cmake")
include(AddExample)
include(LinkPyTorchToTarget)
include(CreateLibWithTests)

#
# Homing Pigeon: Library sources
#
set(LIB_HOPI_SRCS
        environments/Environment.h
        environments/EnvType.h
        environments/FrozenLakeEnv.h environments/FrozenLakeEnv.cpp
        environments/DisentangleSpritesEnv.h environments/DisentangleSpritesEnv.cpp
        environments/MazeEnv.h environments/MazeEnv.cpp
        environments/GraphEnv.h environments/GraphEnv.cpp
        algorithms/inference/VMP.h algorithms/inference/VMP.cpp
        algorithms/planning/EvaluationType.h
        algorithms/planning/NodeSelectionType.h
        algorithms/planning/PropagationType.h
        algorithms/planning/ActionSelectionType.h
        algorithms/planning/MCTS.h algorithms/planning/MCTS.cpp
        algorithms/planning/MCTSConfig.cpp algorithms/planning/MCTSConfig.h
        algorithms/planning/MCTSNodeData.cpp algorithms/planning/MCTSNodeData.h
        distributions/ActiveTransition.h distributions/ActiveTransition.cpp
        distributions/Transition.h distributions/Transition.cpp
        distributions/Categorical.h distributions/Categorical.cpp
        distributions/Distribution.h
        distributions/DistributionType.h
        distributions/Dirichlet.cpp distributions/Dirichlet.h
        graphs/FactorGraph.h graphs/FactorGraph.cpp
        graphs/GraphViz.cpp graphs/GraphViz.h
        nodes/VarNode.h nodes/VarNode.cpp
        nodes/FactorNode.h nodes/FactorNode.cpp
        nodes/VarNodeType.h
        nodes/CategoricalNode.h nodes/CategoricalNode.cpp
        nodes/TransitionNode.h nodes/TransitionNode.cpp
        nodes/ActiveTransitionNode.h nodes/ActiveTransitionNode.cpp
        nodes/DirichletNode.cpp nodes/DirichletNode.h
        iterators/AdjacentFactorsIter.h iterators/AdjacentFactorsIter.cpp
        iterators/HiddenVarIter.h iterators/HiddenVarIter.cpp
        iterators/ObservedVarIter.h iterators/ObservedVarIter.cpp
        math/Ops.cpp math/Ops.h
        api/API.cpp api/API.h
        api/Aliases.h
        zoo/Human.cpp zoo/Human.h
        zoo/BTAI.cpp zoo/BTAI.h)

# Add prefix to all files in HOPI_SRCS
list(TRANSFORM LIB_HOPI_SRCS PREPEND "${LIB_HOPI_ROOT}/${HOPI_SRCS_PATH}")

#
# Homing Pigeon: Unit tests sources
#
set(HOPI_TEST_SRCS
        algorithms/TestMCTS.cpp
        algorithms/TestVMP.cpp
        distributions/TestActiveTransition.cpp
        distributions/TestTransition.cpp
        distributions/TestCategorical.cpp
        distributions/TestDirichlet.cpp
        environments/TestMazeEnv.cpp
        graphs/TestFactorGraph.cpp
        iterators/TestAdjacentFactorsIter.cpp
        iterators/TestHiddenVarIter.cpp
        iterators/TestObservedVarIter.cpp
        nodes/TestActiveTransitionNode.cpp
        nodes/TestCategoricalNode.cpp
        nodes/TestDirichletNode.cpp
        nodes/TestTransitionNode.cpp
        nodes/TestVarNode.cpp
        math/OpsTest.cpp
        # Helpers and contexts only useful for the unit tests
        contexts/FactorGraphContexts.cpp contexts/FactorGraphContexts.h
        helpers/Files.cpp helpers/Files.h
        helpers/UnitTests.cpp helpers/UnitTests.h)

# Add prefix to all files in HOPI_TEST_SRCS
list(TRANSFORM HOPI_TEST_SRCS PREPEND "${LIB_HOPI_ROOT}/${HOPI_TESTS_PATH}")

#
# Create the hopi library and the associated test
#
create_lib_with_tests(
    CATCH_DIR ${LIB_CATCH_ROOT}
    LIB
        TARGET hopi
        SOURCES ${LIB_HOPI_SRCS}
        PUBLIC_INCLUDE_DIRS ${LIB_HOPI_ROOT}/${HOPI_SRCS_PATH}
    TESTS
        TARGET hopiTest
        SOURCES ${HOPI_TEST_SRCS}
        PUBLIC_INCLUDE_DIRS ${LIB_HOPI_ROOT}/${HOPI_TESTS_PATH}
)

# Link OpenCV to hopi
find_package(OpenCV REQUIRED)
target_include_directories(hopi PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(hopi PUBLIC ${OpenCV_LIBS})

# Link pytorch to hopi
link_pytorch_to_target(
    TARGET hopi
    TORCH_DIR ${LIB_TORCH_ROOT}
)

# Examples of Homing Pigeon
add_example(NAME maze_environment           HOPI_PROJECT_DIR ${HOPI_PROJECT_ROOT})
add_example(NAME torch_tensors              HOPI_PROJECT_DIR ${HOPI_PROJECT_ROOT})
add_example(NAME maze_navigation            HOPI_PROJECT_DIR ${HOPI_PROJECT_ROOT})
add_example(NAME graph_environment          HOPI_PROJECT_DIR ${HOPI_PROJECT_ROOT})
add_example(NAME learning_maze_navigation   HOPI_PROJECT_DIR ${HOPI_PROJECT_ROOT})
add_example(NAME factor_graph_visualisation HOPI_PROJECT_DIR ${HOPI_PROJECT_ROOT})
add_example(NAME deep_learning_mnist        HOPI_PROJECT_DIR ${HOPI_PROJECT_ROOT})
