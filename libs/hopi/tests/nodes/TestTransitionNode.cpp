//
// Created by Theophile Champion on 02/12/2020.
//

#include <iostream>
#include "catch.hpp"
#include "nodes/VarNode.h"
#include "nodes/TransitionNode.h"
#include "distributions/Dirichlet.h"
#include "distributions/Categorical.h"
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

TEST_CASE( "TransitionNode.vfe() returns the proper vfe contribution" ) {
    UnitTests::run([](){
        FactorGraph::setCurrent(nullptr);
        auto fg = FactorGraph::current();
        auto c1 = API::Categorical(Ops::uniform({4}));
        auto t1 = API::Transition(c1, Ops::uniform({2,4}));

        auto poc1 = c1->posterior()->params();
        auto pot1 = t1->posterior()->params();
        auto lpt1 = t1->posterior()->logParams();
        auto prt1 = t1->prior()->logParams();

        auto neg_entropy = dot(pot1, lpt1);
        auto energy = dot(matmul(prt1.permute({1,0}), pot1), poc1); // dot(matmul(prt1.permute({1,0}), pot1), poc1);
        REQUIRE( t1->parent()->vfe() == (neg_entropy - energy).item<double>());
    });
}

TEST_CASE( "TransitionNode returns the correct child and parents" ) {
    UnitTests::run([](){
        auto from   = VarNode::create(VarNodeType::HIDDEN);
        auto to     = VarNode::create(VarNodeType::HIDDEN);
        auto factor = TransitionNode::create(from.get(), to.get());

        REQUIRE( factor->child() == to.get() );
        REQUIRE( factor->parent(0) == from.get() );
        REQUIRE( factor->parent(1) == nullptr );
    });
}

TEST_CASE( "TransitionNode's (child and parent) messages are correct (no Dirichlet prior)" ) {
    UnitTests::run([](){
        FactorGraph::setCurrent(nullptr);
        Tensor evidence = Ops::uniform({2});
        Tensor param1   = Ops::uniform({4});
        Tensor param2   = Ops::uniform({2,4});
        auto fg = FactorGraph::current();
        auto c1 = API::Categorical(param1);
        auto t1 = API::Transition(c1, param2);
        t1->setPosterior(Categorical::create(evidence));
        t1->setType(VarNodeType::OBSERVED);

        Tensor res1 = matmul(param2.log().permute({1,0}), evidence);
        auto m1 = t1->parent()->message(c1);
        REQUIRE( equal(m1, res1) );

        Tensor res2 = matmul(param2.log(), param1);
        auto m2 = t1->parent()->message(t1);
        REQUIRE( equal(m2, res2) );
    });
}

TEST_CASE( "TransitionNode's (child, param and parent) messages are correct (Dirichlet prior)" ) {
    UnitTests::run([](){
        FactorGraph::setCurrent(nullptr);
        Tensor evidence = Ops::uniform({2});
        Tensor param1   = Ops::uniform({4});
        Tensor param2   = Ops::uniform({4,2});
        auto fg = FactorGraph::current();
        auto c1 = API::Categorical(param1);
        auto d1 = API::Dirichlet(param2);
        auto t1 = API::Transition(c1, d1);
        t1->setPosterior(Categorical::create(evidence));
        t1->setType(VarNodeType::OBSERVED);

        Tensor res1 = matmul(Dirichlet::expectedLog(param2), evidence);
        auto m1 = t1->parent()->message(c1);
        REQUIRE( equal(m1, res1) );

        Tensor res2 = matmul(Dirichlet::expectedLog(param2).permute({1,0}), param1);
        auto m2 = t1->parent()->message(t1);
        REQUIRE( equal(m2, res2) );

        Tensor res3 = outer(param1, evidence);
        auto m3 = t1->parent()->message(d1);
        REQUIRE( equal(m3, res3) );
    });
}
