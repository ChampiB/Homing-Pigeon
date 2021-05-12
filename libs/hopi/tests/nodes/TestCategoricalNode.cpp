//
// Created by tmac3 on 02/12/2020.
//

#include <iostream>
#include "catch.hpp"
#include "nodes/VarNode.h"
#include "math/Ops.h"
#include "nodes/CategoricalNode.h"
#include "distributions/Dirichlet.h"
#include "graphs/FactorGraph.h"
#include "api/API.h"
#include "helpers/UnitTests.h"
#include <torch/torch.h>

using namespace hopi::math;
using namespace hopi::api;
using namespace hopi::nodes;
using namespace hopi::graphs;
using namespace hopi::distributions;
using namespace torch;
using namespace tests;

TEST_CASE( "CategoricalNode.vfe() returns the proper vfe contribution" ) {
    UnitTests::run([](){
        FactorGraph::setCurrent(nullptr);
        auto c1 = API::Categorical(Ops::uniformColumnWise({4}));

        auto poc1 = c1->posterior()->params()[0];
        auto lpc1 = c1->posterior()->logParams()[0];
        auto prc1 = c1->prior()->logParams()[0];

        auto neg_entropy = matmul(poc1.permute({1,0}), lpc1)[0][0];
        auto energy = matmul(poc1.permute({1,0}), prc1)[0][0];
        REQUIRE( c1->parent()->vfe() == (neg_entropy - energy).item<double>() );
    });
}

TEST_CASE( "CategoricalNode.name getter and setter works properly" ) {
    UnitTests::run([](){
        auto to     = VarNode::create(VarNodeType::HIDDEN);
        auto factor = CategoricalNode::create(to.get());

        REQUIRE( factor->name().empty() );
        factor->setName("test");
        REQUIRE( factor->name() == "test" );
        factor->setName("abc");
        REQUIRE( factor->name() == "abc" );
    });
}

TEST_CASE( "CategoricalNode returns the correct child and parents" ) {
    UnitTests::run([](){
        auto to     = VarNode::create(VarNodeType::HIDDEN);
        auto factor = CategoricalNode::create(to.get());

        REQUIRE( factor->child() == to.get() );
        REQUIRE( factor->parent(0) == nullptr );
    });
}

TEST_CASE( "CategoricalNode::message throws a run_time error is thrown if the parameter is not the generated node" ) {
    UnitTests::run([](){
        FactorGraph::setCurrent(nullptr);
        auto c1 = VarNode::create(VarNodeType::HIDDEN);
        auto fg = FactorGraph::current();
        auto c2 = API::Categorical(Ops::uniformColumnWise({4}));

        try {
            c2->parent()->message(c1.get());
            REQUIRE( false );
        } catch (const std::runtime_error& error) {
            // Correct
        }
    });
}

TEST_CASE( "CategoricalNode's (child) message is correct (no Dirichlet prior)" ) {
    UnitTests::run([](){
        FactorGraph::setCurrent(nullptr);
        auto fg = FactorGraph::current();
        Tensor  param1 = Ops::uniformColumnWise({4});
        auto c1 = API::Categorical(param1);
        auto m1 = c1->parent()->message(c1);
        REQUIRE( torch::equal(m1[0], param1.log()) );

        FactorGraph::setCurrent(nullptr);
        Tensor param2 = torch::tensor({0.25,0.1,0.4,0.25});
        auto c2 = API::Categorical(param2);
        auto m2 = c2->parent()->message(c2);
        REQUIRE( torch::equal(m2[0], param2.log()) );
    });
}

TEST_CASE( "CategoricalNode's (child) message is correct (Dirichlet prior)" ) {
    UnitTests::run([](){
        FactorGraph::setCurrent(nullptr);
        auto fg = FactorGraph::current();
        auto d1 = API::Dirichlet(Ops::uniformColumnWise({4}));
        auto c1 = API::Categorical(d1);
        auto m1 = c1->parent()->message(c1);
        auto res = Dirichlet::expectedLog(d1->posterior()->params())[0];
        REQUIRE( torch::equal(m1[0], res) );

        FactorGraph::setCurrent(nullptr);
        auto d2 = API::Dirichlet(torch::tensor({0.25,0.1,0.4,0.25}));
        auto c2 = API::Categorical(d2);
        auto m2 = c2->parent()->message(c2);
        res = Dirichlet::expectedLog(d2->posterior()->params())[0];
        REQUIRE( torch::equal(m2[0], res) );
    });
}

TEST_CASE( "CategoricalNode's (parent) message is correct (Dirichlet prior)" ) {
    UnitTests::run([](){
        FactorGraph::setCurrent(nullptr);
        auto fg = FactorGraph::current();
        Tensor param1 = Ops::uniformColumnWise({4});
        auto d1 = API::Dirichlet(param1);
        auto c1 = API::Categorical(d1);
        auto m1 = c1->parent()->message(d1);
        REQUIRE( torch::equal(m1[0], torch::tensor({0.25,0.25,0.25,0.25})) );

        FactorGraph::setCurrent(nullptr);
        auto d2 = API::Dirichlet(torch::tensor({0.25,0.1,0.4,0.25}));
        auto c2 = API::Categorical(d2);
        Tensor param2 = torch::tensor({0.7, 0.01, 0.01, 0.28});
        c2->posterior()->updateParams(param2.view({1, 4, 1}));
        auto m2 = c2->parent()->message(d2);
        auto res = torch::softmax(param2, 0);
        REQUIRE( torch::equal(m2[0], res) );
    });
}
