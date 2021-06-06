//
// Created by Theophile Champion on 02/12/2020.
//

#include <iostream>
#include "catch.hpp"
#include "nodes/VarNode.h"
#include "nodes/CategoricalNode.h"
#include "distributions/Dirichlet.h"
#include "graphs/FactorGraph.h"
#include "math/Ops.h"
#include "helpers/UnitTests.h"
#include "api/API.h"
#include <torch/torch.h>

using namespace hopi::nodes;
using namespace hopi::graphs;
using namespace hopi::distributions;
using namespace hopi::api;
using namespace hopi::math;
using namespace torch;
using namespace tests;

TEST_CASE( "DirichletNode returns the correct child and parents" ) {
    UnitTests::run([](){
        auto to     = VarNode::create(VarNodeType::HIDDEN);
        auto factor = CategoricalNode::create(to.get());

        REQUIRE( factor->child() == to.get() );
        REQUIRE( factor->parent(0) == nullptr );
    });
}

TEST_CASE( "DirichletNode's (child) message is correct" ) {
    UnitTests::run([](){
        FactorGraph::setCurrent(nullptr);
        Tensor param1 = Ops::uniform({4});
        auto c1 = API::Dirichlet(param1);
        auto m1 = c1->parent()->message(c1);
        REQUIRE( equal(m1, param1) );

        FactorGraph::setCurrent(nullptr);
        Tensor param2 = API::tensor({0.25,0.1,0.4,0.25});
        auto c2 = API::Dirichlet(param2);
        auto m2 = c2->parent()->message(c2);
        REQUIRE( equal(m2, param2) );
    });
}

TEST_CASE( "DirichletNode.vfe() returns the proper vfe contribution" ) {
    UnitTests::run([](){
        FactorGraph::setCurrent(nullptr);
        auto c1 = API::Dirichlet(Ops::uniform({2,4}));
        REQUIRE( c1->parent()->vfe() == Approx(0.0) ); // TODO check that

        FactorGraph::setCurrent(nullptr);
        auto c2 = API::Dirichlet(API::tensor({1.0,10.0,20.0,5.0}));
        REQUIRE( c2->parent()->vfe() == Approx(-7.10543e-15) ); // TODO check that
    });
}
