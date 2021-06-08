//
// Created by Theophile Champion on 02/12/2020.
//

#include <iostream>
#include "catch.hpp"
#include "iterators/ObservedVarIter.h"
#include "contexts/FactorGraphContexts.h"
#include "graphs/FactorGraph.h"
#include "helpers/UnitTests.h"

using namespace hopi::iterators;
using namespace tests;

TEST_CASE( "ObservedVarIter do not crash when no more hidden variables are available" ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context3();
        auto it = ObservedVarIter(fg.get());

        for (int i = 0; i < 10; ++i) {
            ++it;
        }
    });
}

TEST_CASE( "ObservedVarIter return nullptr when no more hidden variables are available" ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context3();
        auto it = ObservedVarIter(fg.get());

        REQUIRE( *it != nullptr );
        ++it;
        REQUIRE( *it != nullptr );
        ++it;
        REQUIRE( *it != nullptr );
        ++it;
        REQUIRE( *it == nullptr );
    });
}

TEST_CASE( "ObservedVarIter return the correct hidden variable (==)" ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context3();
        auto it = ObservedVarIter(fg.get());

        REQUIRE( *it == fg->node(1) );
        ++it;
        REQUIRE( *it == fg->node(4) );
        ++it;
        REQUIRE( *it == fg->node(5) );
    });
}

TEST_CASE( "ObservedVarIter return the correct hidden variable (!=)" ) {
    UnitTests::run([](){
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
    });
}

TEST_CASE( "ObservedVarIter properly support copy (=)" ) {
    UnitTests::run([](){
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
    });
}
