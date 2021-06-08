//
// Created by Theophile Champion on 02/12/2020.
//

#include "catch.hpp"
#include "nodes/VarNode.h"
#include "nodes/CategoricalNode.h"
#include "distributions/Categorical.h"
#include "distributions/Distribution.h"
#include "helpers/UnitTests.h"
#include "math/Ops.h"
#include <iostream>

using namespace hopi::distributions;
using namespace hopi::nodes;
using namespace hopi::math;
using namespace tests;
using namespace torch;

TEST_CASE( "VarNode's constructor correctly set the node's type" ) {
    UnitTests::run([](){
        auto n1 = VarNode::create(VarNodeType::HIDDEN);
        auto n2 = VarNode::create(VarNodeType::OBSERVED);

        REQUIRE( n1->type() == VarNodeType::HIDDEN );
        REQUIRE( n2->type() == VarNodeType::OBSERVED );
    });
}

TEST_CASE( "VarNode's type setter/getter work properly" ) {
    UnitTests::run([](){
        auto n1 = VarNode::create(VarNodeType::HIDDEN);

        REQUIRE( n1->type() == VarNodeType::HIDDEN );
        n1->setType(VarNodeType::OBSERVED);
        REQUIRE( n1->type() == VarNodeType::OBSERVED );
        n1->setType(VarNodeType::HIDDEN);
        REQUIRE( n1->type() == VarNodeType::HIDDEN );
    });
}

TEST_CASE( "VarNode's G setter/getter work properly" ) {
    UnitTests::run([](){
        auto n1 = VarNode::create(VarNodeType::HIDDEN);

        n1->setG(0);
        REQUIRE( n1->g() == 0 );
        n1->setG(-42);
        REQUIRE( n1->g() == -42 );
    });
}

TEST_CASE( "VarNode's action setter/getter work properly" ) {
    UnitTests::run([](){
        auto n1 = VarNode::create(VarNodeType::HIDDEN);

        n1->setAction(0);
        REQUIRE( n1->action() == 0 );
        n1->setAction(42);
        REQUIRE( n1->action() == 42 );
    });
}

TEST_CASE( "VarNode's n getter/incrementer work properly" ) {
    UnitTests::run([](){
        auto n1 = VarNode::create(VarNodeType::HIDDEN);

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
    });
}

TEST_CASE( "VarNode's prior getter/setter work properly" ) {
    UnitTests::run([](){
        auto d1 = Categorical::create(Ops::uniform({2}));
        Distribution *res = d1.get();
        auto n1 = VarNode::create(VarNodeType::HIDDEN);

        REQUIRE( n1->prior() == nullptr );
        n1->setPrior(std::move(d1));
        REQUIRE( n1->prior() == res );
    });
}

TEST_CASE( "VarNode's posterior getter/setter work properly" ) {
    UnitTests::run([](){
        auto d1 = Categorical::create(Ops::uniform({2}));
        Distribution *res = d1.get();
        auto n1 = VarNode::create(VarNodeType::HIDDEN);

        REQUIRE( n1->posterior() == nullptr );
        n1->setPosterior(std::move(d1));
        REQUIRE( n1->posterior() == res );
    });
}

TEST_CASE( "VarNode's biased getter/setter work properly" ) {
    UnitTests::run([](){
        auto d1 = Categorical::create(Ops::uniform({2}));
        Distribution *res = d1.get();
        auto n1 = VarNode::create(VarNodeType::HIDDEN);

        REQUIRE( n1->biased() == nullptr );
        n1->setBiased(std::move(d1));
        REQUIRE( n1->biased() == res );
    });
}

TEST_CASE( "VarNode's parent getter/setter work properly" ) {
    UnitTests::run([](){
        auto n1 = VarNode::create(VarNodeType::HIDDEN);
        auto n2 = CategoricalNode::create(n1.get());
        auto n3 = VarNode::create(VarNodeType::HIDDEN);

        REQUIRE( n3->parent() == nullptr );
        n3->setParent(n2.get());
        REQUIRE( n3->parent() == n2.get() );
    });
}

TEST_CASE( "VarNode's name getter/setter work properly" ) {
    UnitTests::run([](){
        auto n1 = VarNode::create(VarNodeType::HIDDEN);

        REQUIRE( n1->name().empty() );
        n1->setName("s1");
        REQUIRE( n1->name() == "s1" );
    });
}

TEST_CASE( "VarNode properly allows child addition and retrieval" ) {
    UnitTests::run([](){
        auto n1 = VarNode::create(VarNodeType::HIDDEN);
        auto f1 = CategoricalNode::create(n1.get());
        auto n2 = VarNode::create(VarNodeType::HIDDEN);
        auto f2 = CategoricalNode::create(n2.get());
        auto n3 = VarNode::create(VarNodeType::HIDDEN);

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
    });
}

TEST_CASE( "VarNode.removeNullChildren removes all null children keep the others untouched" ) {
    UnitTests::run([](){
        auto n1 = VarNode::create(VarNodeType::HIDDEN);
        auto f1 = CategoricalNode::create(n1.get());
        auto n2 = VarNode::create(VarNodeType::HIDDEN);
        auto f2 = CategoricalNode::create(n2.get());
        auto n3 = VarNode::create(VarNodeType::HIDDEN);

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
    });
}

TEST_CASE( "VarNode.disconnectChild set the corresponding child to null" ) {
    UnitTests::run([](){
        auto n0 = VarNode::create(VarNodeType::HIDDEN);
        auto f0 = CategoricalNode::create(n0.get());
        auto n1 = VarNode::create(VarNodeType::HIDDEN);
        auto f1 = CategoricalNode::create(n1.get());
        auto n2 = VarNode::create(VarNodeType::HIDDEN);
        auto f2 = CategoricalNode::create(n2.get());
        auto n3 = VarNode::create(VarNodeType::HIDDEN);

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
    });
}
