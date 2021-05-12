//
// Created by tmac3 on 02/12/2020.
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
        auto c1 = API::Categorical(Ops::uniformColumnWise({4}));
        auto t1 = API::Transition(c1, Ops::uniformColumnWise({2,4}));

        auto poc1 = c1->posterior()->params()[0];
        auto pot1 = t1->posterior()->params()[0];
        auto lpt1 = t1->posterior()->logParams()[0];
        auto prt1 = t1->prior()->logParams()[0];

        auto neg_entropy = matmul(pot1.permute({1,0}), lpt1);
        auto energy = matmul(matmul(pot1.permute({1,0}), prt1), poc1);
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

TEST_CASE( "TransitionNode: A run_time error is thrown if the parameter is an unknown node" ) {
    UnitTests::run([](){
        FactorGraph::setCurrent(nullptr);
        auto t1 = VarNode::create(VarNodeType::HIDDEN);
        auto fg = FactorGraph::current();
        auto c1 = API::Categorical(Ops::uniformColumnWise({4}));
        auto t2 = API::Transition(c1, Ops::uniformColumnWise({2,4}));

        try {
            t2->parent()->message(t1.get());
            REQUIRE( false );
        } catch (const std::runtime_error& error) {
            // Correct
        }
    });
}

TEST_CASE( "TransitionNode's (child and parent) messages are correct (no Dirichlet prior)" ) {
    UnitTests::run([](){
        FactorGraph::setCurrent(nullptr);
        Tensor evidence = Ops::uniformColumnWise({2});
        Tensor param1   = Ops::uniformColumnWise({4});
        Tensor param2   = Ops::uniformColumnWise({2,4});
        auto fg = FactorGraph::current();
        auto c1 = API::Categorical(param1);
        auto t1 = API::Transition(c1, param2);
        t1->setPosterior(Categorical::create(evidence));
        t1->setType(VarNodeType::OBSERVED);

        Tensor res1 = matmul(param2.log().permute({1,0}), evidence);
        auto m1 = t1->parent()->message(c1);
        REQUIRE( torch::equal(m1[0], res1) );

        Tensor res2 = matmul(param2.log(), param1);
        auto m2 = t1->parent()->message(t1);
        REQUIRE( torch::equal(m2[0], res2) );
    });
}

TEST_CASE( "TransitionNode's (child, param and parent) messages are correct (Dirichlet prior)" ) {
    UnitTests::run([](){
        FactorGraph::setCurrent(nullptr);
        Tensor evidence = Ops::uniformColumnWise({2});
        Tensor param1   = Ops::uniformColumnWise({4});
        Tensor param2   = Ops::uniformColumnWise({1, 2, 4});
        auto fg = FactorGraph::current();
        auto c1 = API::Categorical(param1);
        auto d1 = API::Dirichlet(param2[0]);
        auto t1 = API::Transition(c1, d1);
        t1->setPosterior(Categorical::create(evidence));
        t1->setType(VarNodeType::OBSERVED);

        Tensor res1 = matmul(Dirichlet::expectedLog(param2)[0].permute({1, 0}), evidence);
        auto m1 = t1->parent()->message(c1);
        REQUIRE( torch::equal(m1[0], res1) );

        Tensor res2 = matmul(Dirichlet::expectedLog(param2)[0], param1);
        auto m2 = t1->parent()->message(t1);
        REQUIRE( torch::equal(m2[0], res2) );

        Tensor res3 = matmul(evidence, param1.permute({1,0}));
        auto m3 = t1->parent()->message(d1);
        REQUIRE( torch::equal(m3[0], res3) );
    });
}
