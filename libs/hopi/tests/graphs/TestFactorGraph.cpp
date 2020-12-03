//
// Created by tmac3 on 02/12/2020.
//

#include "catch.hpp"
#include "nodes/VarNode.h"
#include "nodes/FactorNode.h"
#include "nodes/CategoricalNode.h"
#include "graphs/FactorGraph.h"
#include "distributions/Categorical.h"
#include "distributions/Transition.h"
#include "distributions/ActiveTransition.h"
#include "helpers/Files.h"
#include "contexts/FactorGraphContexts.h"
#include <Eigen/Dense>
#include <iostream>

using namespace hopi::distributions;
using namespace hopi::graphs;
using namespace hopi::nodes;
using namespace tests;
using namespace Eigen;

TEST_CASE( "FactorGraph.getNodes returns a vector containing all vars of the graph" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    FactorGraph::setCurrent(nullptr);
    std::shared_ptr<FactorGraph> fg = FactorGraph::current();
    MatrixXd U0 = MatrixXd::Constant(5, 1, 1.0 / 5);
    VarNode *a0 = Categorical::create(U0);
    a0->setType(VarNodeType::OBSERVED);
    VarNode *a1 = Categorical::create(U0);
    VarNode *a2 = Categorical::create(U0);
    a2->setType(VarNodeType::OBSERVED);
    VarNode *a3 = Categorical::create(U0);
    VarNode *a4 = Categorical::create(U0);
    auto vec = fg->getNodes();
    REQUIRE( vec.size() == 5 );
    REQUIRE( vec[0] == a0 );
    REQUIRE( vec[1] == a1 );
    REQUIRE( vec[2] == a2 );
    REQUIRE( vec[3] == a3 );
    REQUIRE( vec[4] == a4 );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "FactorGraph's tree_root getter/setter work properly" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    std::unique_ptr<VarNode> n1 = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    FactorGraph::setCurrent(nullptr);
    auto fg = FactorGraph::current();

    REQUIRE( fg->treeRoot() == nullptr );
    fg->setTreeRoot(n1.get());
    REQUIRE( fg->treeRoot() == n1.get() );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "FactorGraph allows addition and retrieval of factors" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto n1   = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    auto f1   = std::make_unique<CategoricalNode>(n1.get());
    auto res1 = f1.get();
    auto n2   = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    auto f2   = std::make_unique<CategoricalNode>(n2.get());
    auto res2 = f2.get();
    auto n3   = std::make_unique<VarNode>(VarNodeType::OBSERVED);
    auto f3   = std::make_unique<CategoricalNode>(n3.get());
    auto res3 = f3.get();
    FactorGraph::setCurrent(nullptr);
    auto fg   = FactorGraph::current();

    REQUIRE( fg->factors() == 0 );

    fg->addFactor(std::move(f1));
    REQUIRE( fg->factors() == 1 );
    REQUIRE( fg->factor(0) == res1 );

    fg->addFactor(std::move(f2));
    REQUIRE( fg->factors() == 2 );
    REQUIRE( fg->factor(0) == res1 );
    REQUIRE( fg->factor(1) == res2 );

    fg->addFactor(std::move(f3));
    REQUIRE( fg->factors() == 3 );
    REQUIRE( fg->factor(0) == res1 );
    REQUIRE( fg->factor(1) == res2 );
    REQUIRE( fg->factor(2) == res3 );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "FactorGraph allows addition and retrieval of nodes" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    FactorGraph::setCurrent(nullptr);
    auto n1   = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    auto res1 = n1.get();
    auto n2   = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    auto res2 = n2.get();
    auto n3   = std::make_unique<VarNode>(VarNodeType::OBSERVED);
    auto res3 = n3.get();
    auto fg   = FactorGraph::current();

    REQUIRE( fg->nHiddenVar() == 0 );
    REQUIRE( fg->nObservedVar() == 0 );
    REQUIRE( fg->nodes() == 0 );

    fg->addNode(std::move(n1));
    REQUIRE( fg->nHiddenVar() == 1 );
    REQUIRE( fg->nObservedVar() == 0 );
    REQUIRE( fg->nodes() == 1 );
    REQUIRE( fg->node(0) == res1 );

    fg->addNode(std::move(n2));
    REQUIRE( fg->nHiddenVar() == 2 );
    REQUIRE( fg->nObservedVar() == 0 );
    REQUIRE( fg->nodes() == 2 );
    REQUIRE( fg->node(0) == res1 );
    REQUIRE( fg->node(1) == res2 );

    fg->addNode(std::move(n3));
    REQUIRE( fg->nHiddenVar() == 2 );
    REQUIRE( fg->nObservedVar() == 1 );
    REQUIRE( fg->nodes() == 3 );
    REQUIRE( fg->node(0) == res1 );
    REQUIRE( fg->node(1) == res2 );
    REQUIRE( fg->node(2) == res3 );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Building a FactorGraph using 'create' functions properly connect nodes" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    FactorGraph::setCurrent(nullptr);
    std::shared_ptr<FactorGraph> fg = FactorGraph::current();

    /**
     ** Create the model's parameters.
     **/
    MatrixXd U0 = MatrixXd::Constant(5, 1, 1.0 / 5);
    MatrixXd D0 = MatrixXd::Constant(3,  1, 1.0 / 3);

    int A_size = 9 * 3;
    MatrixXd A = MatrixXd::Constant(9, 3, 1.0 / A_size);

    int B_size = 3 * 3;
    MatrixXd B_idle  = MatrixXd::Constant(3, 3, 1.0 / B_size);
    MatrixXd B_up    = MatrixXd::Constant(3, 3, 1.0 / B_size);
    MatrixXd B_down  = MatrixXd::Constant(3, 3, 1.0 / B_size);
    MatrixXd B_right = MatrixXd::Constant(3, 3, 1.0 / B_size);
    MatrixXd B_left  = MatrixXd::Constant(3, 3, 1.0 / B_size);
    std::vector<MatrixXd> B = {B_up, B_down, B_left, B_right, B_idle};

    /**
     ** Create the generative model.
     **/
    VarNode *a0 = Categorical::create(U0);
    REQUIRE( fg->nodes() == 1 );
    REQUIRE( fg->factors() == 1 );
    REQUIRE( a0->parent() == fg->factor(0) ); // parent of a0 is P_a0
    REQUIRE( fg->factor(0)->child() == a0 );  // child of P_a0 is a0

    VarNode *s0 = Categorical::create(D0);
    REQUIRE( fg->nodes() == 2 );
    REQUIRE( fg->factors() == 2 );
    REQUIRE( s0->parent() == fg->factor(1) ); // parent of s0 is P_s0
    REQUIRE( fg->factor(1)->child() == s0 );  // child of P_s0 is s0

    VarNode *o0 = Transition::create(s0, A);
    REQUIRE( fg->nodes() == 3 );
    REQUIRE( fg->factors() == 3 );
    REQUIRE( o0->parent() == fg->factor(2) );      // parent of o0 is P_o0
    REQUIRE( fg->factor(2)->child() == o0 );       // child of P_o0 is o0
    REQUIRE( *s0->firstChild() == fg->factor(2) ); // child of s0 is P_o0
    REQUIRE( fg->factor(2)->parent(0) == s0 );     // parent of P_o0 is s0

    VarNode *s1 = ActiveTransition::create(s0, a0, B);
    REQUIRE( fg->nodes() == 4 );
    REQUIRE( fg->factors() == 4 );
    REQUIRE( s1->parent() == fg->factor(3) );          // parent of s1 is P_s1
    REQUIRE( fg->factor(3)->child() == s1 );           // child of P_s1 is s1
    REQUIRE( *(++s0->firstChild()) == fg->factor(3) ); // child of s0 is P_s1
    REQUIRE( fg->factor(3)->parent(0) == s0 );         // parent of P_s1 is s0
    REQUIRE( *a0->firstChild() == fg->factor(3) );     // child of a0 is P_s1
    REQUIRE( fg->factor(3)->parent(1) == a0 );         // parent of P_s1 is a0
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "FactorGraph's oneHot returns correct one hot vectors" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd m1 = FactorGraph::oneHot(2, 1);
    REQUIRE( m1(0, 0) == 0 );
    REQUIRE( m1(1, 0) == 1 );

    MatrixXd m2 = FactorGraph::oneHot(5, 0);
    REQUIRE( m2(0, 0) == 1 );
    REQUIRE( m2(1, 0) == 0 );
    REQUIRE( m2(2, 0) == 0 );
    REQUIRE( m2(3, 0) == 0 );
    REQUIRE( m2(4, 0) == 0 );

    MatrixXd m3 = FactorGraph::oneHot(3, 2);
    REQUIRE( m3(0, 0) == 0 );
    REQUIRE( m3(1, 0) == 0 );
    REQUIRE( m3(2, 0) == 1 );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "FactorGraph's loadEvidence properly load evidence into observed variables" ) {
    FactorGraph::setCurrent(nullptr);
    auto n1 = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    auto s1 = n1.get();
    auto n2 = std::make_unique<VarNode>(VarNodeType::OBSERVED);
    n2->setName("o0");
    auto s2 = n2.get();
    auto n3 = std::make_unique<VarNode>(VarNodeType::OBSERVED);
    n3->setName("o1");
    auto s3 = n3.get();
    auto fg = FactorGraph::current();

    fg->addNode(std::move(n1));
    fg->addNode(std::move(n2));
    fg->addNode(std::move(n3));

    REQUIRE( s1->posterior() == nullptr );
    REQUIRE( s2->posterior() == nullptr );
    REQUIRE( s3->posterior() == nullptr );

    fg->loadEvidence(10, Files::getEvidencePath("1.evi"));

    REQUIRE( s1->posterior() == nullptr );

    REQUIRE( s2->posterior() != nullptr );
    REQUIRE( s2->posterior()->type() == DistributionType::CATEGORICAL );
    auto c2 = dynamic_cast<Categorical*>(s2->posterior());
    REQUIRE( c2->p(0) == 0 );
    REQUIRE( c2->p(1) == 0 );
    REQUIRE( c2->p(2) == 0 );
    REQUIRE( c2->p(3) == 0 );
    REQUIRE( c2->p(4) == 0 );
    REQUIRE( c2->p(5) == 0 );
    REQUIRE( c2->p(6) == 0 );
    REQUIRE( c2->p(7) == 0 );
    REQUIRE( c2->p(8) == 1 );
    REQUIRE( c2->p(9) == 0 );

    REQUIRE( s3->posterior() != nullptr );
    REQUIRE( s3->posterior()->type() == DistributionType::CATEGORICAL );
    auto c3 = dynamic_cast<Categorical*>(s3->posterior());
    REQUIRE( c3->p(0) == 0 );
    REQUIRE( c3->p(1) == 0 );
    REQUIRE( c3->p(2) == 0 );
    REQUIRE( c3->p(3) == 0 );
    REQUIRE( c3->p(4) == 0 );
    REQUIRE( c3->p(5) == 0 );
    REQUIRE( c3->p(6) == 0 );
    REQUIRE( c3->p(7) == 0 );
    REQUIRE( c3->p(8) == 0 );
    REQUIRE( c3->p(9) == 1 );
}

TEST_CASE( "FactorGraph.integrate properly update the factor graph" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    std::shared_ptr<FactorGraph> fg = FactorGraphContexts::context2();

    int A_size = 9 * 3;
    MatrixXd A = MatrixXd::Constant(9, 3, 1.0 / A_size);
    int B_size = 3 * 3;
    MatrixXd B_0 = MatrixXd::Constant(3, 3, 1.0 / B_size);
    MatrixXd B_1 = MatrixXd::Constant(3, 3, 1.0 / B_size);
    std::vector<MatrixXd> B = {B_0, B_1};

    auto root = fg->treeRoot();

    // First expansion
    auto sI = Transition::create(root, B_1);
    sI->setAction(1);
    auto oI = Transition::create(sI, A);

    // Second expansion
    auto sJ = Transition::create(root, B_0);
    sJ->setAction(0);
    auto oJ = Transition::create(sJ, A);

    // Third expansion
    auto sK = Transition::create(sJ, B_0);
    sK->setAction(0);
    auto oK = Transition::create(sK, A);

    fg->integrate(0, fg->oneHot(9, 2), A, B);
    auto new_root = fg->treeRoot();
    REQUIRE( new_root != root );  // Root have been updated
    REQUIRE( new_root->prior()->type() == DistributionType::ACTIVE_TRANSITION ); // The type of prior is now an active transition
    REQUIRE( new_root->parent()->parent(0) == root ); // The first parent of new_root is the (old) root node
    auto d1 = new_root->parent()->parent(1)->prior();
    REQUIRE( d1->type() == DistributionType::CATEGORICAL); // The second parent of new_root has a categorical prior
    auto c1 = dynamic_cast<Categorical*>(d1);
    REQUIRE( c1->cardinality() == 2); // and its cardinality is 5
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "FactorGraph.removeNullNodes properly remove null variables from the factor graph" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    FactorGraph::setCurrent(nullptr);
    std::shared_ptr<FactorGraph> fg = std::make_shared<FactorGraph>();

    fg->addNode(nullptr);
    auto v1 = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    auto p1 = v1.get();
    fg->addNode(std::move(v1));
    fg->addNode(nullptr);
    fg->addNode(nullptr);
    fg->addNode(nullptr);
    auto v2 = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    auto p2 = v2.get();
    fg->addNode(std::move(v2));
    auto v3 = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    auto p3 = v3.get();
    fg->addNode(std::move(v3));
    fg->addNode(nullptr);
    REQUIRE( fg->nodes() == 8 );
    fg->removeNullNodes();
    REQUIRE( fg->nodes() == 3 );
    REQUIRE( fg->node(0) == p1 );
    REQUIRE( fg->node(1) == p2 );
    REQUIRE( fg->node(2) == p3 );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "FactorGraph.removeNullFactors properly remove null factors from the factor graph" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    FactorGraph::setCurrent(nullptr);
    std::shared_ptr<FactorGraph> fg = std::make_shared<FactorGraph>();

    fg->addFactor(nullptr);
    auto v1 = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    auto f1 = std::make_unique<CategoricalNode>(v1.get());
    auto p1 = f1.get();
    fg->addFactor(std::move(f1));
    fg->addFactor(nullptr);
    fg->addFactor(nullptr);
    fg->addFactor(nullptr);
    auto v2 = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    auto f2 = std::make_unique<CategoricalNode>(v2.get());
    auto p2 = f2.get();
    fg->addFactor(std::move(f2));
    auto v3 = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    auto f3 = std::make_unique<CategoricalNode>(v3.get());
    auto p3 = f3.get();
    fg->addFactor(std::move(f3));
    fg->addFactor(nullptr);
    REQUIRE( fg->factors() == 8 );
    fg->removeNullFactors();
    REQUIRE( fg->factors() == 3 );
    REQUIRE( fg->factor(0) == p1 );
    REQUIRE( fg->factor(1) == p2 );
    REQUIRE( fg->factor(2) == p3 );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "FactorGraph.removeBranch properly cut off branches of the tree" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    std::shared_ptr<FactorGraph> fg = FactorGraphContexts::context2();
    int A_size = 9 * 3;
    MatrixXd A = MatrixXd::Constant(9, 3, 1.0 / A_size);
    int B_size = 3 * 3;
    MatrixXd B_0 = MatrixXd::Constant(3, 3, 1.0 / B_size);
    MatrixXd B_1 = MatrixXd::Constant(3, 3, 1.0 / B_size);
    auto root = fg->treeRoot();

    REQUIRE( fg->factors() == 5 );
    REQUIRE( fg->nodes() == 5 );
    REQUIRE( root->lastChild() - root->firstChild() == 1 );

    auto s0   = Transition::create(root, B_0);
    auto s00  = Transition::create(s0,   B_0);
    auto s000 = Transition::create(s00,  B_0);
    auto s001 = Transition::create(s00,  B_1);
    auto s1   = Transition::create(root, B_1);
    auto s11  = Transition::create(s1,   B_1);
    auto s110 = Transition::create(s11,  B_0);
    auto s111 = Transition::create(s11,  B_1);

    REQUIRE( fg->factors() == 13 );
    REQUIRE( fg->nodes() == 13 );
    REQUIRE( root->lastChild() - root->firstChild() == 3 );

    fg->removeBranch(s0->parent());
    root->removeNullChildren();
    fg->removeNullNodes();
    fg->removeNullFactors();
    REQUIRE( fg->factors() == 9 );
    REQUIRE( fg->nodes() == 9 );
    REQUIRE( root->lastChild() - root->firstChild() == 2 );
    REQUIRE( fg->node(5) == s1 );
    REQUIRE( fg->node(6) == s11 );
    REQUIRE( fg->node(7) == s110 );
    REQUIRE( fg->node(8) == s111 );

    fg->removeBranch(s1->parent());
    root->removeNullChildren();
    fg->removeNullNodes();
    fg->removeNullFactors();
    REQUIRE( fg->factors() == 5 );
    REQUIRE( fg->nodes() == 5 );
    REQUIRE( root->lastChild() - root->firstChild() == 1 );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}
