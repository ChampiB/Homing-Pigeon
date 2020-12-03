//
// Created by tmac3 on 02/12/2020.
//

#include <iostream>
#include "catch.hpp"
#include "iterators/ObservedVarIter.h"
#include "contexts/FactorGraphContexts.h"
#include "graphs/FactorGraph.h"

using namespace hopi::iterators;
using namespace tests;

TEST_CASE( "ObservedVarIter do not crash when no more hidden variables are available" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto fg = FactorGraphContexts::context3();
    auto it = ObservedVarIter(fg.get());

    for (int i = 0; i < 10; ++i) {
        ++it;
    }
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "ObservedVarIter return nullptr when no more hidden variables are available" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto fg = FactorGraphContexts::context3();
    auto it = ObservedVarIter(fg.get());

    REQUIRE( *it != nullptr );
    ++it;
    REQUIRE( *it != nullptr );
    ++it;
    REQUIRE( *it != nullptr );
    ++it;
    REQUIRE( *it == nullptr );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "ObservedVarIter return the correct hidden variable (==)" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto fg = FactorGraphContexts::context3();
    auto it = ObservedVarIter(fg.get());

    REQUIRE( *it == fg->node(1) );
    ++it;
    REQUIRE( *it == fg->node(4) );
    ++it;
    REQUIRE( *it == fg->node(5) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "ObservedVarIter return the correct hidden variable (!=)" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto fg = FactorGraphContexts::context3();
    auto it = ObservedVarIter(fg.get());

    REQUIRE( *it != fg->node(4) );
    REQUIRE( *it != fg->node(5) );
    ++it;
    REQUIRE( *it != fg->node(1) );
    REQUIRE( *it != fg->node(5) );
    ++it;
    REQUIRE( *it != fg->node(1) );
    REQUIRE( *it != fg->node(4) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "ObservedVarIter properly support copy (=)" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto fg = FactorGraphContexts::context3();
    auto it = ObservedVarIter(fg.get());

    REQUIRE( *it == fg->node(1) );
    ++it;
    auto it2 = it;
    REQUIRE( *it == fg->node(4) );
    REQUIRE( *it2 == fg->node(4) );

    ++it;
    REQUIRE( *it == fg->node(5) );
    ++it2;
    REQUIRE( *it2 == fg->node(5) );

    ++it;
    REQUIRE( *it == nullptr );
    ++it2;
    REQUIRE( *it2 == nullptr );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}
