//
// Created by tmac3 on 02/12/2020.
//

#include <iostream>
#include "catch.hpp"
#include "nodes/VarNode.h"
#include "nodes/ActiveTransitionNode.h"
#include "graphs/FactorGraph.h"
#include "distributions/Dirichlet.h"
#include "distributions/Categorical.h"
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

TEST_CASE( "ActiveTransitionNode.vfe() returns the proper vfe contribution" ) {
    UnitTests::run([](){
        FactorGraph::setCurrent(nullptr);
        auto fg = FactorGraph::current();
        auto c1 = API::Categorical(Ops::uniformColumnWise({4}));
        auto c2 = API::Categorical(Ops::uniformColumnWise({2}));
        auto t1 = API::ActiveTransition(c1, c2, Ops::uniformColumnWise({2, 4, 4}));

        auto poc1 = c1->posterior()->params();
        auto poc2 = c2->posterior()->params();
        auto pot1 = t1->posterior()->params().permute({1,0});
        auto lpt1 = t1->posterior()->logParams();
        auto prt1 = t1->prior()->logParams();

        Tensor neg_entropy = matmul(pot1, lpt1);
        Tensor energy = matmul(poc2[0][0], matmul(matmul(pot1, prt1[0]), poc1)) \
                      + matmul(poc2[1][0], matmul(matmul(pot1, prt1[1]), poc1));
        REQUIRE( c1->parent()->vfe() == (neg_entropy - energy).item<double>());
    });
}

TEST_CASE( "ActiveTransitionNode returns the correct child and parents" ) {
    UnitTests::run([](){
        auto from   = VarNode::create(VarNodeType::HIDDEN);
        auto action = VarNode::create(VarNodeType::HIDDEN);
        auto to     = VarNode::create(VarNodeType::HIDDEN);
        auto factor = ActiveTransitionNode::create(from.get(), action.get(), to.get());

        REQUIRE( factor->child() == to.get() );
        REQUIRE( factor->parent(0) == from.get() );
        REQUIRE( factor->parent(1) == action.get() );
        REQUIRE( factor->parent(2) == nullptr );
    });
}

TEST_CASE( "ActiveTransitionNode: A run_time error is thrown if the parameter is an unknown node" ) {
    UnitTests::run([](){
        FactorGraph::setCurrent(nullptr);
        Tensor param = Ops::uniformColumnWise({4});
        Tensor param2 = Ops::uniformColumnWise({2, 4, 4});
        auto t1 = VarNode::create(VarNodeType::HIDDEN);
        auto fg = FactorGraph::current();
        auto c1 = API::Categorical(param);
        auto c2 = API::Categorical(param);
        auto t2 = API::ActiveTransition(c1, c2, param2);

        try {
            t2->parent()->message(t1.get());
            REQUIRE( false );
        } catch (const std::runtime_error& error) {
            // Correct
        }
    });
}

TEST_CASE( "ActiveTransitionNode's (to, from and action) messages are correct (no Dirichlet prior)" ) {
    UnitTests::run([](){
        FactorGraph::setCurrent(nullptr);
        Tensor U = Ops::uniformColumnWise({2});
        Tensor D = Ops::uniformColumnWise({4});
        Tensor evidence = Ops::uniformColumnWise({2});
        Tensor B = Ops::uniformColumnWise({2, 2, 4});
        auto fg = FactorGraph::current();
        auto c1 = API::Categorical(D);
        auto c2 = API::Categorical(U);
        auto t1 = API::ActiveTransition(c1, c2, B);
        t1->setPosterior(Categorical::create(evidence));
        t1->setType(VarNodeType::OBSERVED);

        Tensor res1 = U[0][0] * matmul(B[0].log(), D) \
                    + U[1][0] * matmul(B[1].log(), D);
        auto m1 = t1->parent()->message(t1);
        REQUIRE( m1[0].size(1) == 1 );
        REQUIRE( m1[0].size(0) == 2 );
        REQUIRE( torch::equal(m1[0], res1) );

        Tensor res2 = U[0][0] * matmul(B[0].log().permute({1,0}), evidence) \
                    + U[1][0] * matmul(B[1].log().permute({1,0}), evidence);
        auto m2 = t1->parent()->message(c1);
        REQUIRE( m2[0].size(1) == 1 );
        REQUIRE( m2[0].size(0) == 4 );
        REQUIRE( torch::equal(m2[0], res2) );

        auto m3 = t1->parent()->message(c2);
        REQUIRE( m3[0].size(1) == 1 );
        REQUIRE( m3[0].size(0) == 2 );
        REQUIRE( torch::equal(m2[0], res2) );

        REQUIRE( torch::equal(m3[0][0][0], matmul(matmul(evidence.permute({1,0}), B[0].log()), D)[0][0]) );
        REQUIRE( torch::equal(m3[0][1][0], matmul(matmul(evidence.permute({1,0}), B[1].log()), D)[0][0]) );
    });
}

TEST_CASE( "ActiveTransitionNode's (to, from, param and action) messages are correct (Dirichlet prior)" ) {
    UnitTests::run([](){
        FactorGraph::setCurrent(nullptr);
        Tensor U = Ops::uniformColumnWise({2});
        Tensor D = Ops::uniformColumnWise({4});
        Tensor evidence = Ops::uniformColumnWise({2});
        Tensor B = Ops::uniformColumnWise({2, 2, 4});
        auto fg = FactorGraph::current();
        auto c1 = API::Categorical(D);
        auto c2 = API::Categorical(U);
        auto d1 = API::Dirichlet(B);
        auto t1 = API::ActiveTransition(c1, c2, d1);
        t1->setPosterior(Categorical::create(evidence));
        t1->setType(VarNodeType::OBSERVED);

        Tensor res1 = U[0][0] * matmul(Dirichlet::expectedLog(B)[0], D) \
                    + U[1][0] * matmul(Dirichlet::expectedLog(B)[1],D);
        auto m1 = t1->parent()->message(t1);
        REQUIRE( m1[0].size(1) == 1 );
        REQUIRE( m1[0].size(0) == 2 );
        REQUIRE( torch::equal(m1, res1) );

        Tensor res2 = U[0][0] * Dirichlet::expectedLog(B)[0].permute({1,0}) * evidence \
                    + U[1][0] * Dirichlet::expectedLog(B)[1].permute({1,0}) * evidence;
        auto m2 = t1->parent()->message(c1);
        REQUIRE( m2[0].size(1) == 1 );
        REQUIRE( m2[0].size(0) == 4 );
        REQUIRE( torch::equal(m2[0], res2) );

        auto m3 = t1->parent()->message(c2);
        REQUIRE( m3[0].size(1) == 1 );
        REQUIRE( m3[0].size(0) == 2 );
        REQUIRE( m3[0][0][0].item<double>() == matmul(matmul(evidence.permute({1,0}), Dirichlet::expectedLog(B)[0]), D)[0][0].item<double>() );
        REQUIRE( m3[0][1][0].item<double>() == matmul(matmul(evidence.permute({1,0}), Dirichlet::expectedLog(B)[1]), D)[0][0].item<double>() );

        Tensor res4 = torch::zeros({2, 2, 4});
        for (int i = 0; i < res4.size(0); ++i) {
            res4[i] = matmul(evidence, D.permute({1,0})) * U[i][0];
        }
        auto m4 = t1->parent()->message(d1);
        REQUIRE( m4.size(0) == 2 );
        REQUIRE( m4.size(2) == 4 );
        REQUIRE( m4.size(1) == 2 );
        REQUIRE( torch::equal(m4,res4) );
    });
}
