//
// Created by tmac3 on 02/12/2020.
//

#include "catch.hpp"
#include "nodes/VarNode.h"
#include "nodes/CategoricalNode.h"
#include "distributions/Categorical.h"
#include "distributions/Distribution.h"
#include <Eigen/Dense>
#include <iostream>

using namespace hopi::distributions;
using namespace hopi::nodes;
using namespace Eigen;

TEST_CASE( "VarNode's constructor correctly set the node's type" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto n1 = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    auto n2 = std::make_unique<VarNode>(VarNodeType::OBSERVED);

    REQUIRE( n1->type() == VarNodeType::HIDDEN );
    REQUIRE( n2->type() == VarNodeType::OBSERVED );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "VarNode's type setter/getter work properly" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto n1 = std::make_unique<VarNode>(VarNodeType::HIDDEN);

    REQUIRE( n1->type() == VarNodeType::HIDDEN );
    n1->setType(VarNodeType::OBSERVED);
    REQUIRE( n1->type() == VarNodeType::OBSERVED );
    n1->setType(VarNodeType::HIDDEN);
    REQUIRE( n1->type() == VarNodeType::HIDDEN );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "VarNode's G setter/getter work properly" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto n1 = std::make_unique<VarNode>(VarNodeType::HIDDEN);

    n1->setG(0);
    REQUIRE( n1->g() == 0 );
    n1->setG(-42);
    REQUIRE( n1->g() == -42 );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "VarNode's action setter/getter work properly" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto n1 = std::make_unique<VarNode>(VarNodeType::HIDDEN);

    n1->setAction(0);
    REQUIRE( n1->action() == 0 );
    n1->setAction(42);
    REQUIRE( n1->action() == 42 );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "VarNode's n getter/incrementer work properly" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto n1 = std::make_unique<VarNode>(VarNodeType::HIDDEN);

    REQUIRE( n1->n() == 0 );
    n1->incrementN();
    REQUIRE( n1->n() == 1 );
    n1->incrementN();
    REQUIRE( n1->n() == 2 );
    n1->incrementN();
    REQUIRE( n1->n() == 3 );
    n1->incrementN();
    n1->incrementN();
    n1->incrementN();
    n1->incrementN();
    REQUIRE( n1->n() == 7 );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "VarNode's prior getter/setter work properly" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd param = MatrixXd(2,1);
    std::unique_ptr<Distribution> d1 = std::make_unique<Categorical>(param);
    Distribution *res = d1.get();
    auto n1 = std::make_unique<VarNode>(VarNodeType::HIDDEN);

    REQUIRE( n1->prior() == nullptr );
    n1->setPrior(std::move(d1));
    REQUIRE( n1->prior() == res );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "VarNode's posterior getter/setter work properly" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd param = MatrixXd(2,1);
    std::unique_ptr<Distribution> d1 = std::make_unique<Categorical>(param);
    Distribution *res = d1.get();
    auto n1 = std::make_unique<VarNode>(VarNodeType::HIDDEN);

    REQUIRE( n1->posterior() == nullptr );
    n1->setPosterior(std::move(d1));
    REQUIRE( n1->posterior() == res );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "VarNode's biased getter/setter work properly" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd param = MatrixXd(2,1);
    std::unique_ptr<Distribution> d1 = std::make_unique<Categorical>(param);
    Distribution *res = d1.get();
    auto n1 = std::make_unique<VarNode>(VarNodeType::HIDDEN);

    REQUIRE( n1->biased() == nullptr );
    n1->setBiased(std::move(d1));
    REQUIRE( n1->biased() == res );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "VarNode's parent getter/setter work properly" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto n1 = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    auto n2 = std::make_unique<CategoricalNode>(n1.get());
    auto n3 = std::make_unique<VarNode>(VarNodeType::HIDDEN);

    REQUIRE( n3->parent() == nullptr );
    n3->setParent(n2.get());
    REQUIRE( n3->parent() == n2.get() );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "VarNode's name getter/setter work properly" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto n1 = std::make_unique<VarNode>(VarNodeType::HIDDEN);

    REQUIRE( n1->name().empty() );
    n1->setName("s1");
    REQUIRE( n1->name() == "s1" );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "VarNode properly allows child addition and retrieval" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto n1 = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    auto f1 = std::make_unique<CategoricalNode>(n1.get());
    auto n2 = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    auto f2 = std::make_unique<CategoricalNode>(n2.get());
    auto n3 = std::make_unique<VarNode>(VarNodeType::HIDDEN);

    REQUIRE( n3->nChildren() == 0 );
    n3->addChild(f1.get());
    REQUIRE( n3->nChildren() == 1 );
    REQUIRE( *n3->firstChild() == f1.get() );
    REQUIRE( *(++n3->firstChild()) == *n3->lastChild() );
    n3->addChild(f2.get());
    REQUIRE( n3->nChildren() == 2 );
    REQUIRE( *n3->firstChild() == f1.get() );
    REQUIRE( *(++n3->firstChild()) == f2.get() );
    REQUIRE( *(++(++n3->firstChild())) == *n3->lastChild() );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "VarNode.removeNullChildren removes all null children keep the others untouched" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto n1 = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    auto f1 = std::make_unique<CategoricalNode>(n1.get());
    auto n2 = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    auto f2 = std::make_unique<CategoricalNode>(n2.get());
    auto n3 = std::make_unique<VarNode>(VarNodeType::HIDDEN);

    n3->addChild(nullptr);
    n3->addChild(f1.get());
    n3->addChild(nullptr);
    n3->addChild(nullptr);
    n3->addChild(nullptr);
    n3->addChild(f2.get());
    n3->addChild(nullptr);
    REQUIRE( n3->nChildren() == 7 );
    n3->removeNullChildren();
    REQUIRE( n3->nChildren() == 2 );
    REQUIRE( *n3->firstChild() == f1.get() );
    REQUIRE( *(++n3->firstChild()) == f2.get() );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "VarNode.disconnectChild set the corresponding child to null" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto n0 = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    auto f0 = std::make_unique<CategoricalNode>(n0.get());
    auto n1 = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    auto f1 = std::make_unique<CategoricalNode>(n1.get());
    auto n2 = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    auto f2 = std::make_unique<CategoricalNode>(n2.get());
    auto n3 = std::make_unique<VarNode>(VarNodeType::HIDDEN);

    n3->addChild(f0.get());
    n3->addChild(f1.get());
    n3->addChild(f2.get());
    REQUIRE( n3->nChildren() == 3 );
    REQUIRE( *n3->firstChild() == f0.get() );
    REQUIRE( *(++n3->firstChild()) == f1.get() );
    REQUIRE( *(++(++n3->firstChild())) == f2.get() );

    n3->disconnectChild(f1.get());
    REQUIRE( n3->nChildren() == 3 );
    REQUIRE( *n3->firstChild() == f0.get() );
    REQUIRE( *(++n3->firstChild()) == nullptr );
    REQUIRE( *(++(++n3->firstChild())) == f2.get() );

    n3->disconnectChild(f2.get());
    REQUIRE( n3->nChildren() == 3 );
    REQUIRE( *n3->firstChild() == f0.get() );
    REQUIRE( *(++n3->firstChild()) == nullptr );
    REQUIRE( *(++(++n3->firstChild())) == nullptr );

    n3->disconnectChild(f0.get());
    REQUIRE( n3->nChildren() == 3 );
    REQUIRE( *n3->firstChild() == nullptr );
    REQUIRE( *(++n3->firstChild()) == nullptr );
    REQUIRE( *(++(++n3->firstChild())) == nullptr );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}
