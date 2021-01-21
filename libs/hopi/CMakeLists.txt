project(HomingPigeonLibrary VERSION 1.0.0 LANGUAGES CXX)

#
# Homing Pigeon: Library
#
set(HOPI_SRCS
        srcs/environments/Environment.h
        srcs/environments/MazeEnv.h srcs/environments/MazeEnv.cpp
        srcs/algorithms/AlgoTree.h srcs/algorithms/AlgoTree.cpp
        srcs/algorithms/AlgoVMP.h srcs/algorithms/AlgoVMP.cpp
        srcs/algorithms/EvaluationType.h
        srcs/algorithms/NodeSelectionType.h
        srcs/distributions/ActiveTransition.h srcs/distributions/ActiveTransition.cpp
        srcs/distributions/Transition.h srcs/distributions/Transition.cpp
        srcs/distributions/Categorical.h srcs/distributions/Categorical.cpp
        srcs/distributions/Distribution.h
        srcs/distributions/DistributionType.h
        srcs/graphs/FactorGraph.h srcs/graphs/FactorGraph.cpp
        srcs/nodes/VarNode.h srcs/nodes/VarNode.cpp
        srcs/nodes/FactorNode.h srcs/nodes/FactorNode.cpp
        srcs/nodes/VarNodeType.h
        srcs/nodes/VarNodeAttr.h
        srcs/nodes/CategoricalNode.h srcs/nodes/CategoricalNode.cpp
        srcs/nodes/TransitionNode.h srcs/nodes/TransitionNode.cpp
        srcs/nodes/ActiveTransitionNode.h srcs/nodes/ActiveTransitionNode.cpp
        srcs/graphs/FactorGraph.h srcs/graphs/FactorGraph.cpp
        srcs/iterators/AdjacentFactorsIter.h srcs/iterators/AdjacentFactorsIter.cpp
        srcs/iterators/HiddenVarIter.h srcs/iterators/HiddenVarIter.cpp
        srcs/iterators/ObservedVarIter.h srcs/iterators/ObservedVarIter.cpp
        srcs/nodes/DirichletNode.cpp srcs/nodes/DirichletNode.h
        srcs/distributions/Dirichlet.cpp srcs/distributions/Dirichlet.h
        srcs/math/Functions.cpp srcs/math/Functions.h)

add_library(hopi STATIC ${HOPI_SRCS})
target_include_directories(hopi PUBLIC ${HOMING_PIGEON_ROOT}/libs/hopi/srcs)
target_include_directories(hopi PUBLIC ${HOMING_PIGEON_ROOT}/libs/eigen)

#
# Homing Pigeon: Unit tests
#
set (TEST_HOPI_SRCS
        tests/algorithms/TestAlgoTree.cpp
        tests/algorithms/TestAlgoVMP.cpp
        tests/distributions/TestActiveTransition.cpp
        tests/distributions/TestTransition.cpp
        tests/distributions/TestCategorical.cpp
        tests/distributions/TestDirichlet.cpp
        tests/environments/TestMazeEnv.cpp
        tests/graphs/TestFactorGraph.cpp
        tests/iterators/TestAdjacentFactorsIter.cpp
        tests/iterators/TestHiddenVarIter.cpp
        tests/iterators/TestObservedVarIter.cpp
        tests/nodes/TestActiveTransitionNode.cpp
        tests/nodes/TestCategoricalNode.cpp
        tests/nodes/TestDirichletNode.cpp
        tests/nodes/TestTransitionNode.cpp
        tests/nodes/TestVarNode.cpp
        tests/math/FunctionsTest.cpp
        tests/contexts/FactorGraphContexts.cpp tests/contexts/FactorGraphContexts.h
        tests/helpers/Files.cpp tests/helpers/Files.h)

add_unit_tests(LIB_NAME hopi TEST_FILES ${TEST_HOPI_SRCS})
