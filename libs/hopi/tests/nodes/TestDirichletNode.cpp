//
// Created by tmac3 on 02/12/2020.
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

TEST_CASE( "DirichletNode::message throws a run_time error is thrown if the parameter is not the generated node" ) {
    UnitTests::run([](){
        FactorGraph::setCurrent(nullptr);
        auto c1 = VarNode::create(VarNodeType::HIDDEN);
        auto fg = FactorGraph::current();
        auto c2 = API::Dirichlet(Ops::uniformColumnWise({4,1}));

        try {
            c2->parent()->message(c1.get());
            REQUIRE( false );
        } catch (const std::runtime_error& error) {
            // Correct
        }
    });
}

TEST_CASE( "DirichletNode's (child) message is correct" ) {
    UnitTests::run([](){
        FactorGraph::setCurrent(nullptr);
        Tensor param1 = Ops::uniformColumnWise({4, 1});
        auto c1 = API::Dirichlet(param1);
        auto m1 = c1->parent()->message(c1);
        REQUIRE( torch::equal(m1[0], param1) );

        FactorGraph::setCurrent(nullptr);
        Tensor param2 = torch::tensor({0.25,0.1,0.4,0.25});
        auto c2 = API::Dirichlet(param2);
        auto m2 = c2->parent()->message(c2);
        REQUIRE( torch::equal(m2[0], param2) );
    });
}

TEST_CASE( "DirichletNode.vfe() returns the proper vfe contribution" ) {
    UnitTests::run([](){
        FactorGraph::setCurrent(nullptr);
        auto c1 = API::Dirichlet(Ops::uniformColumnWise({2, 4}));
        REQUIRE( c1->parent()->vfe() == Approx(-7.8540401042) );

        FactorGraph::setCurrent(nullptr);
        auto c2 = API::Dirichlet(torch::tensor({1,10,20,5}));
        REQUIRE( c2->parent()->vfe() == Approx(-0.7301998275) );
    });
}
