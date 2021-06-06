//
// Created by Theophile Champion on 02/12/2020.
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
        auto c1 = API::Categorical(Ops::uniform({4}));
        auto c2 = API::Categorical(Ops::uniform({2}));
        auto t1 = API::ActiveTransition(c1, c2, Ops::uniform({4,4,2}));

        auto poc1 = c1->posterior()->params();
        auto poc2 = c2->posterior()->params();
        auto pot1 = t1->posterior()->params();
        auto lpt1 = t1->posterior()->logParams();
        auto prt1 = t1->prior()->logParams();

        Tensor neg_entropy = matmul(pot1, lpt1);
        Tensor energy = Ops::average(prt1, poc2, {2});
        energy = Ops::average(energy, poc1, {1});
        energy = Ops::average(energy, pot1, {0});
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

TEST_CASE( "ActiveTransitionNode's (to, from and action) messages are correct (no Dirichlet prior)" ) {
    UnitTests::run([](){
        FactorGraph::setCurrent(nullptr);
        Tensor U = Ops::uniform({4});
        Tensor D = Ops::uniform({2});
        Tensor evidence = Ops::uniform({2});
        Tensor B = Ops::uniform({2,2,4});
        auto fg = FactorGraph::current();
        auto c1 = API::Categorical(D);
        auto c2 = API::Categorical(U);
        auto t1 = API::ActiveTransition(c1, c2, B);
        t1->setPosterior(Categorical::create(evidence));
        t1->setType(VarNodeType::OBSERVED);

        Tensor res1 = Ops::average(B.log(), U, {2});
        res1 = Ops::average(res1, D, {1});
        auto m1 = t1->parent()->message(t1);
        REQUIRE( equal(m1, res1) );

        Tensor res2 = Ops::average(B.log(), U, {2});
        res2 = Ops::average(res2, evidence, {0});
        auto m2 = t1->parent()->message(c1);
        REQUIRE( equal(m2, res2) );

        Tensor res3 = Ops::average(B.log(), D, {1});
        res3 = Ops::average(res3, evidence, {0});
        auto m3 = t1->parent()->message(c2);
        REQUIRE( equal(m3, res3) );
    });
}

TEST_CASE( "ActiveTransitionNode's (to, from, param and action) messages are correct (Dirichlet prior)" ) {
    UnitTests::run([](){
        FactorGraph::setCurrent(nullptr);
        Tensor U = Ops::uniform({4});
        Tensor D = Ops::uniform({2});
        Tensor evidence = Ops::uniform({2});
        Tensor B = Ops::uniform({2,4,2});
        auto fg = FactorGraph::current();
        auto c1 = API::Categorical(D);
        auto c2 = API::Categorical(U);
        auto d1 = API::Dirichlet(B);
        auto t1 = API::ActiveTransition(c1, c2, d1);
        t1->setPosterior(Categorical::create(evidence));
        t1->setType(VarNodeType::OBSERVED);

        Tensor res1 = Ops::average(Dirichlet::expectedLog(B), U, {1});
        res1 = Ops::average(res1, D, {0});
        auto m1 = t1->parent()->message(t1);
        REQUIRE( equal(m1, res1) );

        Tensor res2 = Ops::average(Dirichlet::expectedLog(B), evidence, {2});
        res2 = Ops::average(res2, U, {1});
        auto m2 = t1->parent()->message(c1);
        REQUIRE( equal(m2, res2) );

        Tensor res3 = Ops::average(Dirichlet::expectedLog(B), evidence, {2});
        res3 = Ops::average(res3, D, {0});
        auto m3 = t1->parent()->message(c2);
        REQUIRE( equal(m3, res3) );

        Tensor D_hat = API::tensor({0.1,0.5,0.4});
        Tensor U_hat = API::tensor({0.1,0.7,0.2});
        c1->setPosterior(Categorical::create(D_hat));
        c2->setPosterior(Categorical::create(U_hat));
        Tensor res4 = Ops::outer_tensor_product({&D_hat,&U_hat,&evidence});
        auto m4 = t1->parent()->message(d1);
        REQUIRE( equal(m4, res4) );
    });
}
