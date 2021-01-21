//
// Created by tmac3 on 02/12/2020.
//

#include <iostream>
#include "catch.hpp"
#include "nodes/VarNode.h"
#include "nodes/ActiveTransitionNode.h"
#include "graphs/FactorGraph.h"
#include "distributions/Distribution.h"
#include "distributions/Dirichlet.h"
#include "distributions/Categorical.h"
#include "distributions/ActiveTransition.h"
#include <Eigen/Dense>

using namespace hopi::nodes;
using namespace hopi::graphs;
using namespace hopi::distributions;
using namespace Eigen;

TEST_CASE( "ActiveTransitionNode.vfe() returns the proper vfe contribution" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    FactorGraph::setCurrent(nullptr);
    MatrixXd param1 = MatrixXd::Constant(4, 1, 0.25);
    MatrixXd param2 = MatrixXd::Constant(2, 1, 0.5);
    std::vector<MatrixXd> param3 {
            MatrixXd::Constant(4, 4, 0.25),
            MatrixXd::Constant(4, 4, 0.25)
    };
    auto fg = FactorGraph::current();
    auto c1 = Categorical::create(param1);
    auto c2 = Categorical::create(param2);
    auto t1 = ActiveTransition::create(c1, c2, param3);

    auto poc1 = c1->posterior()->params()[0];
    auto poc2 = c2->posterior()->params()[0];
    auto pot1 = t1->posterior()->params()[0];
    auto lpt1 = t1->posterior()->logParams()[0];
    auto prt1 = t1->prior()->logParams();

    double neg_entropy = (pot1.transpose() * lpt1)(0, 0);
    double energy  = poc2(0, 0) * (pot1.transpose() * prt1[0] * poc1)(0, 0);
           energy += poc2(1, 0) * (pot1.transpose() * prt1[1] * poc1)(0, 0);
    REQUIRE( c1->parent()->vfe() == neg_entropy - energy);
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "ActiveTransitionNode returns the correct child and parents" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto from   = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    auto action = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    auto to     = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    auto factor = std::make_unique<ActiveTransitionNode>(from.get(), action.get(), to.get());

    REQUIRE( factor->child() == to.get() );
    REQUIRE( factor->parent(0) == from.get() );
    REQUIRE( factor->parent(1) == action.get() );
    REQUIRE( factor->parent(2) == nullptr );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "ActiveTransitionNode: A run_time error is thrown if the parameter is an unknown node" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    FactorGraph::setCurrent(nullptr);
    MatrixXd param = MatrixXd::Constant(4, 1, 0.25);
    std::vector<MatrixXd> param2 {
            MatrixXd::Constant(4, 4, 0.25),
            MatrixXd::Constant(4, 4, 0.25)
    };
    auto t1 = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    auto fg = FactorGraph::current();
    auto c1 = Categorical::create(param);
    auto c2 = Categorical::create(param);
    auto t2 = ActiveTransition::create(c1, c2, param2);

    try {
        t2->parent()->message(t1.get());
        REQUIRE( false );
    } catch (const std::runtime_error& error) {
        // Correct
    }
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "ActiveTransitionNode's (to, from and action) messages are correct (no Dirichlet prior)" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    FactorGraph::setCurrent(nullptr);
    MatrixXd U = MatrixXd::Constant(2, 1, 0.5);
    MatrixXd D = MatrixXd::Constant(4, 1, 0.25);
    MatrixXd evidence = MatrixXd::Constant(2, 1, 0.5);
    std::vector<MatrixXd> B {
            MatrixXd::Constant(2, 4, 0.25),
            MatrixXd::Constant(2, 4, 0.25)
    };
    auto fg = FactorGraph::current();
    auto c1 = Categorical::create(D);
    auto c2 = Categorical::create(U);
    auto t1 = ActiveTransition::create(c1, c2, B);
    t1->setPosterior(std::make_unique<Categorical>(evidence));
    t1->setType(VarNodeType::OBSERVED);

    MatrixXd res11 = B[0].array().log();
    res11 = U(0 , 0) * res11 * D;
    MatrixXd res12 = B[1].array().log();
    res12 = U(1 , 0) * res12 * D;
    MatrixXd res1 = res11 + res12;
    auto m1 = t1->parent()->message(t1);
    REQUIRE( m1[0].cols() == 1 );
    REQUIRE( m1[0].rows() == 2 );
    REQUIRE( m1[0](0, 0) == res1(0, 0) );
    REQUIRE( m1[0](1, 0) == res1(1, 0) );

    MatrixXd res21 = B[0].array().log();
    res21 = U(0 , 0) * res21.transpose() * evidence;
    MatrixXd res22 = B[1].array().log();
    res22 = U(1 , 0) * res22.transpose() * evidence;
    MatrixXd res2 = res21 + res22;
    auto m2 = t1->parent()->message(c1);
    REQUIRE( m2[0].cols() == 1 );
    REQUIRE( m2[0].rows() == 4 );
    REQUIRE( m2[0](0, 0) == res2(0, 0) );
    REQUIRE( m2[0](1, 0) == res2(1, 0) );
    REQUIRE( m2[0](2, 0) == res2(2, 0) );
    REQUIRE( m2[0](3, 0) == res2(3, 0) );

    MatrixXd res31 = B[0].array().log();
    MatrixXd res32 = B[1].array().log();
    auto m3 = t1->parent()->message(c2);
    REQUIRE( m3[0].cols() == 1 );
    REQUIRE( m3[0].rows() == 2 );
    REQUIRE( m3[0](0, 0) == (evidence.transpose() * res31 * D)(0, 0) );
    REQUIRE( m3[0](1, 0) == (evidence.transpose() * res32 * D)(0, 0) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "ActiveTransitionNode's (to, from, param and action) messages are correct (Dirichlet prior)" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    FactorGraph::setCurrent(nullptr);
    MatrixXd U = MatrixXd::Constant(2, 1, 0.5);
    MatrixXd D = MatrixXd::Constant(4, 1, 0.25);
    MatrixXd evidence = MatrixXd::Constant(2, 1, 0.5);
    std::vector<MatrixXd> B {
            MatrixXd::Constant(2, 4, 0.25),
            MatrixXd::Constant(2, 4, 0.25)
    };
    auto fg = FactorGraph::current();
    auto c1 = Categorical::create(D);
    auto c2 = Categorical::create(U);
    auto d1 = Dirichlet::create(B);
    auto t1 = ActiveTransition::create(c1, c2, d1);
    t1->setPosterior(std::make_unique<Categorical>(evidence));
    t1->setType(VarNodeType::OBSERVED);

    MatrixXd res11 = Dirichlet::expectedLog(B)[0];
    res11 = U(0 , 0) * res11 * D;
    MatrixXd res12 = Dirichlet::expectedLog(B)[1];
    res12 = U(1 , 0) * res12 * D;
    MatrixXd res1 = res11 + res12;
    auto m1 = t1->parent()->message(t1);
    REQUIRE( m1[0].cols() == 1 );
    REQUIRE( m1[0].rows() == 2 );
    REQUIRE( m1[0](0, 0) == res1(0, 0) );
    REQUIRE( m1[0](1, 0) == res1(1, 0) );

    MatrixXd res21 = Dirichlet::expectedLog(B)[0];
    res21 = U(0 , 0) * res21.transpose() * evidence;
    MatrixXd res22 = Dirichlet::expectedLog(B)[1];
    res22 = U(1 , 0) * res22.transpose() * evidence;
    MatrixXd res2 = res21 + res22;
    auto m2 = t1->parent()->message(c1);
    REQUIRE( m2[0].cols() == 1 );
    REQUIRE( m2[0].rows() == 4 );
    REQUIRE( m2[0](0, 0) == res2(0, 0) );
    REQUIRE( m2[0](1, 0) == res2(1, 0) );
    REQUIRE( m2[0](2, 0) == res2(2, 0) );
    REQUIRE( m2[0](3, 0) == res2(3, 0) );

    MatrixXd res31 = Dirichlet::expectedLog(B)[0];
    MatrixXd res32 = Dirichlet::expectedLog(B)[1];
    auto m3 = t1->parent()->message(c2);
    REQUIRE( m3[0].cols() == 1 );
    REQUIRE( m3[0].rows() == 2 );
    REQUIRE( m3[0](0, 0) == (evidence.transpose() * res31 * D)(0, 0) );
    REQUIRE( m3[0](1, 0) == (evidence.transpose() * res32 * D)(0, 0) );

    std::vector<MatrixXd> res4 {
        MatrixXd::Zero(2, 4),
        MatrixXd::Zero(2, 4)
    };
    for (int i = 0; i < res4.size(); ++i) {
        res4[i] = evidence * D.transpose() * U(i, 0);
    }
    auto m4 = t1->parent()->message(d1);
    REQUIRE( m4.size() == 2 );
    REQUIRE( m4[0].cols() == 4 );
    REQUIRE( m4[0].rows() == 2 );
    REQUIRE( m4[1].cols() == 4 );
    REQUIRE( m4[1].rows() == 2 );
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 4; ++k) {
                REQUIRE( m4[i](j, k) == res4[i](j, k) );
            }
        }
    }
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}
