//
// Created by tmac3 on 02/12/2020.
//

#include "catch.hpp"
#include "nodes/VarNode.h"
#include "nodes/CategoricalNode.h"
#include "graphs/FactorGraph.h"
#include "distributions/Categorical.h"
#include "helpers/Files.h"
#include "math/Ops.h"
#include "api/API.h"
#include "contexts/FactorGraphContexts.h"
#include <torch/torch.h>
#include "helpers/UnitTests.h"

using namespace hopi::distributions;
using namespace hopi::graphs;
using namespace hopi::nodes;
using namespace hopi::math;
using namespace hopi::api;
using namespace tests;
using namespace torch;

TEST_CASE( "FactorGraph.getNodes returns a vector containing all vars of the graph" ) {
    UnitTests::run([](){
        FactorGraph::setCurrent(nullptr);
        std::shared_ptr<FactorGraph> fg = FactorGraph::current();
        Tensor U0 = Ops::uniformColumnWise({5});
        VarNode *a0 = API::Categorical(U0);
        a0->setType(VarNodeType::OBSERVED);
        VarNode *a1 = API::Categorical(U0);
        VarNode *a2 = API::Categorical(U0);
        a2->setType(VarNodeType::OBSERVED);
        VarNode *a3 = API::Categorical(U0);
        VarNode *a4 = API::Categorical(U0);
        auto vec = fg->getNodes();
        REQUIRE( vec.size() == 5 );
        REQUIRE( vec[0] == a0 );
        REQUIRE( vec[1] == a1 );
        REQUIRE( vec[2] == a2 );
        REQUIRE( vec[3] == a3 );
        REQUIRE( vec[4] == a4 );
    });
}

TEST_CASE( "FactorGraph's tree_root getter/setter work properly" ) {
    UnitTests::run([](){
        auto n1 = VarNode::create(VarNodeType::HIDDEN);
        FactorGraph::setCurrent(nullptr);
        auto fg = FactorGraph::current();

        REQUIRE( fg->treeRoot() == nullptr );
        fg->setTreeRoot(n1.get());
        REQUIRE( fg->treeRoot() == n1.get() );
    });
}

TEST_CASE( "FactorGraph allows addition and retrieval of factors" ) {
    UnitTests::run([](){
        auto n1   = VarNode::create(VarNodeType::HIDDEN);
        auto f1   = CategoricalNode::create(n1.get());
        auto res1 = f1.get();
        auto n2   = VarNode::create(VarNodeType::HIDDEN);
        auto f2   = CategoricalNode::create(n2.get());
        auto res2 = f2.get();
        auto n3   = VarNode::create(VarNodeType::OBSERVED);
        auto f3   = CategoricalNode::create(n3.get());
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
    });
}

TEST_CASE( "FactorGraph allows addition and retrieval of nodes" ) {
    UnitTests::run([](){
        FactorGraph::setCurrent(nullptr);
        auto n1   = VarNode::create(VarNodeType::HIDDEN);
        auto res1 = n1.get();
        auto n2   = VarNode::create(VarNodeType::HIDDEN);
        auto res2 = n2.get();
        auto n3   = VarNode::create(VarNodeType::OBSERVED);
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
    });
}

TEST_CASE( "Building a FactorGraph using 'create' functions properly connect nodes (no learning)" ) {
    UnitTests::run([](){
        FactorGraph::setCurrent(nullptr);
        auto fg = FactorGraph::current();

        /**
         ** Create the model's parameters.
         **/
        Tensor U0 = Ops::uniformColumnWise({5});
        Tensor D0 = Ops::uniformColumnWise({3});
        Tensor A  = Ops::uniformColumnWise({9,3});
        Tensor B  = Ops::uniformColumnWise({5,3,3});

        /**
         ** Create the generative model.
         **/
        VarNode *a0 = API::Categorical(U0);
        REQUIRE( fg->nodes() == 1 );
        REQUIRE( fg->factors() == 1 );
        REQUIRE( a0->parent() == fg->factor(0) ); // parent of a0 is P_a0
        REQUIRE( fg->factor(0)->child() == a0 );  // child of P_a0 is a0

        VarNode *s0 = API::Categorical(D0);
        REQUIRE( fg->nodes() == 2 );
        REQUIRE( fg->factors() == 2 );
        REQUIRE( s0->parent() == fg->factor(1) ); // parent of s0 is P_s0
        REQUIRE( fg->factor(1)->child() == s0 );  // child of P_s0 is s0

        VarNode *o0 = API::Transition(s0, A);
        REQUIRE( fg->nodes() == 3 );
        REQUIRE( fg->factors() == 3 );
        REQUIRE( o0->parent() == fg->factor(2) );      // parent of o0 is P_o0
        REQUIRE( fg->factor(2)->child() == o0 );       // child of P_o0 is o0
        REQUIRE( *s0->firstChild() == fg->factor(2) ); // child of s0 is P_o0
        REQUIRE( fg->factor(2)->parent(0) == s0 );     // parent of P_o0 is s0

        VarNode *s1 = API::ActiveTransition(s0, a0, B);
        REQUIRE( fg->nodes() == 4 );
        REQUIRE( fg->factors() == 4 );
        REQUIRE( s1->parent() == fg->factor(3) );          // parent of s1 is P_s1
        REQUIRE( fg->factor(3)->child() == s1 );           // child of P_s1 is s1
        REQUIRE( *(++s0->firstChild()) == fg->factor(3) ); // child of s0 is P_s1
        REQUIRE( fg->factor(3)->parent(0) == s0 );         // parent of P_s1 is s0
        REQUIRE( *a0->firstChild() == fg->factor(3) );     // child of a0 is P_s1
        REQUIRE( fg->factor(3)->parent(1) == a0 );         // parent of P_s1 is a0
    });
}

TEST_CASE( "Building a FactorGraph using 'create' functions properly connect nodes (Dirichlet learning)" ) {
    UnitTests::run([](){
        FactorGraph::setCurrent(nullptr);
        auto fg = FactorGraph::current();
        int actions = 2;
        int observations = 2;
        int states = 2;

        /**
         ** Create the model's parameters.
         **/
        Tensor theta_U = torch::ones({actions, 1});
        Tensor theta_A = torch::ones({observations, states});
        Tensor theta_D = torch::ones({states, 1});
        Tensor theta_B = torch::ones({actions,states,states});

        /**
         ** Create the generative model.
         **/
        VarNode *U = API::Dirichlet(theta_U);
        REQUIRE( fg->nodes() == 1 );
        REQUIRE( fg->factors() == 1 );
        REQUIRE( U->parent() == fg->factor(0) ); // parent of U is P_U
        REQUIRE( fg->factor(0)->child() == U );  // child of P_U is U

        VarNode *A = API::Dirichlet(theta_A);
        REQUIRE( fg->nodes() == 2 );
        REQUIRE( fg->factors() == 2 );
        REQUIRE( A->parent() == fg->factor(1) ); // parent of A is P_A
        REQUIRE( fg->factor(1)->child() == A );  // child of P_A is A

        VarNode *B = API::Dirichlet(theta_B);
        REQUIRE( fg->nodes() == 3 );
        REQUIRE( fg->factors() == 3 );
        REQUIRE( B->parent() == fg->factor(2) ); // parent of B is P_B
        REQUIRE( fg->factor(2)->child() == B );  // child of P_B is B

        VarNode *D = API::Dirichlet(theta_D);
        REQUIRE( fg->nodes() == 4 );
        REQUIRE( fg->factors() == 4 );
        REQUIRE( D->parent() == fg->factor(3) ); // parent of D is P_D
        REQUIRE( fg->factor(3)->child() == D );  // child of P_D is D

        VarNode *a0 = API::Categorical(U);
        REQUIRE( fg->nodes() == 5 );
        REQUIRE( fg->factors() == 5 );
        REQUIRE( a0->parent() == fg->factor(4) );     // parent of a0 is P_a0
        REQUIRE( fg->factor(4)->child() == a0 );      // child of P_a0 is a0
        REQUIRE( *U->firstChild() == fg->factor(4) ); // child of U is P_a0
        REQUIRE( fg->factor(4)->parent(0) == U );     // parent of P_a0 is U

        VarNode *s0 = API::Categorical(D);
        REQUIRE( fg->nodes() == 6 );
        REQUIRE( fg->factors() == 6 );
        REQUIRE( s0->parent() == fg->factor(5) );     // parent of s0 is P_s0
        REQUIRE( fg->factor(5)->child() == s0 );      // child of P_s0 is s0
        REQUIRE( *D->firstChild() == fg->factor(5) ); // child of D is P_s0
        REQUIRE( fg->factor(5)->parent(0) == D );     // parent of P_s0 is D

        VarNode *o0 = API::Transition(s0, A);
        REQUIRE( fg->nodes() == 7 );
        REQUIRE( fg->factors() == 7 );
        REQUIRE( o0->parent() == fg->factor(6) );      // parent of o0 is P_o0
        REQUIRE( fg->factor(6)->child() == o0 );       // child of P_o0 is o0
        REQUIRE( *s0->firstChild() == fg->factor(6) ); // child of s0 is P_o0
        REQUIRE( fg->factor(6)->parent(0) == s0 );     // parent of P_o0 is s0
        REQUIRE( *A->firstChild() == fg->factor(6) );  // child of A is P_o0
        REQUIRE( fg->factor(6)->parent(1) == A );      // parent of P_o0 is A

        VarNode *s1 = API::ActiveTransition(s0, a0, B);
        REQUIRE( fg->nodes() == 8 );
        REQUIRE( fg->factors() == 8 );
        REQUIRE( s1->parent() == fg->factor(7) );          // parent of s1 is P_s1
        REQUIRE( fg->factor(7)->child() == s1 );           // child of P_s1 is s1
        REQUIRE( *(++s0->firstChild()) == fg->factor(7) ); // child of s0 is P_s1
        REQUIRE( fg->factor(7)->parent(0) == s0 );         // parent of P_s1 is s0
        REQUIRE( *a0->firstChild() == fg->factor(7) );     // child of a0 is P_s1
        REQUIRE( fg->factor(7)->parent(1) == a0 );         // parent of P_s1 is a0
        REQUIRE( *B->firstChild() == fg->factor(7) );      // child of B is P_s1
        REQUIRE( fg->factor(7)->parent(2) == B );          // parent of P_s1 is B

        VarNode *o1 = API::Transition(s1, A);
        REQUIRE( fg->nodes() == 9 );
        REQUIRE( fg->factors() == 9 );
        REQUIRE( o1->parent() == fg->factor(8) );          // parent of o1 is P_o1
        REQUIRE( fg->factor(8)->child() == o1 );           // child of P_o1 is o1
        REQUIRE( *s1->firstChild() == fg->factor(8) );     // child of s1 is P_o1
        REQUIRE( fg->factor(8)->parent(0) == s1 );         // parent of P_o1 is s1
        REQUIRE( *(++A->firstChild()) == fg->factor(8) );  // child of A is P_o1
        REQUIRE( fg->factor(8)->parent(1) == A );          // parent of P_o1 is A
    });
}

TEST_CASE( "FactorGraph's loadEvidence properly load evidence into observed variables" ) {
    UnitTests::run([](){
        FactorGraph::setCurrent(nullptr);
        auto n1 = VarNode::create(VarNodeType::HIDDEN);
        auto s1 = n1.get();
        auto n2 = VarNode::create(VarNodeType::OBSERVED);
        n2->setName("o0");
        auto s2 = n2.get();
        auto n3 = VarNode::create(VarNodeType::OBSERVED);
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
        REQUIRE( torch::equal(s2->posterior()->params(), Ops::oneHot(10,8)) );

        REQUIRE( s3->posterior() != nullptr );
        REQUIRE( s3->posterior()->type() == DistributionType::CATEGORICAL );
        REQUIRE( torch::equal(s3->posterior()->params(), Ops::oneHot(10,9)) );
    });
}

TEST_CASE( "FactorGraph.integrate properly update the factor graph" ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context2();

        Tensor A = Ops::uniformColumnWise({9, 3});
        Tensor B = Ops::uniformColumnWise({2, 3, 3});

        auto root = fg->treeRoot();

        // First expansion
        auto sI = API::Transition(root, B[1]);
        sI->setAction(1);
        auto oI = API::Transition(sI, A);

        // Second expansion
        auto sJ = API::Transition(root, B[0]);
        sJ->setAction(0);
        auto oJ = API::Transition(sJ, A);

        // Third expansion
        auto sK = API::Transition(sJ, B[0]);
        sK->setAction(0);
        auto oK = API::Transition(sK, A);

        fg->integrate(0, Ops::oneHot(9, 2), A, B);
        auto new_root = fg->treeRoot();
        REQUIRE( new_root != root );  // Root have been updated
        REQUIRE( new_root->prior()->type() == DistributionType::ACTIVE_TRANSITION ); // The type of prior is now an active transition
        REQUIRE( new_root->parent()->parent(0) == root ); // The first parent of new_root is the (old) root node
        auto d1 = new_root->parent()->parent(1)->prior();
        REQUIRE( d1->type() == DistributionType::CATEGORICAL); // The second parent of new_root has a categorical prior
        REQUIRE( d1->params().numel() == 2); // and its cardinality is 5
    });
}

TEST_CASE( "FactorGraph.removeNullNodes properly remove null variables from the factor graph" ) {
    UnitTests::run([](){
        FactorGraph::setCurrent(nullptr);
        auto fg = std::make_shared<FactorGraph>();

        fg->addNode(nullptr);
        auto v1 = VarNode::create(VarNodeType::HIDDEN);
        auto p1 = v1.get();
        fg->addNode(std::move(v1));
        fg->addNode(nullptr);
        fg->addNode(nullptr);
        fg->addNode(nullptr);
        auto v2 = VarNode::create(VarNodeType::HIDDEN);
        auto p2 = v2.get();
        fg->addNode(std::move(v2));
        auto v3 = VarNode::create(VarNodeType::HIDDEN);
        auto p3 = v3.get();
        fg->addNode(std::move(v3));
        fg->addNode(nullptr);
        REQUIRE( fg->nodes() == 8 );
        fg->removeNullNodes();
        REQUIRE( fg->nodes() == 3 );
        REQUIRE( fg->node(0) == p1 );
        REQUIRE( fg->node(1) == p2 );
        REQUIRE( fg->node(2) == p3 );
    });
}

TEST_CASE( "FactorGraph.removeNullFactors properly remove null factors from the factor graph" ) {
    UnitTests::run([](){
        FactorGraph::setCurrent(nullptr);
        std::shared_ptr<FactorGraph> fg = std::make_shared<FactorGraph>();

        fg->addFactor(nullptr);
        auto v1 = VarNode::create(VarNodeType::HIDDEN);
        auto f1 = CategoricalNode::create(v1.get());
        auto p1 = f1.get();
        fg->addFactor(std::move(f1));
        fg->addFactor(nullptr);
        fg->addFactor(nullptr);
        fg->addFactor(nullptr);
        auto v2 = VarNode::create(VarNodeType::HIDDEN);
        auto f2 = CategoricalNode::create(v2.get());
        auto p2 = f2.get();
        fg->addFactor(std::move(f2));
        auto v3 = VarNode::create(VarNodeType::HIDDEN);
        auto f3 = CategoricalNode::create(v3.get());
        auto p3 = f3.get();
        fg->addFactor(std::move(f3));
        fg->addFactor(nullptr);
        REQUIRE( fg->factors() == 8 );
        fg->removeNullFactors();
        REQUIRE( fg->factors() == 3 );
        REQUIRE( fg->factor(0) == p1 );
        REQUIRE( fg->factor(1) == p2 );
        REQUIRE( fg->factor(2) == p3 );
    });
}

TEST_CASE( "FactorGraph.removeBranch properly cut off branches of the tree" ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context2();
        Tensor A = Ops::uniformColumnWise({9, 3});
        Tensor B = Ops::uniformColumnWise({2, 3, 3});
        auto root = fg->treeRoot();

        REQUIRE( fg->factors() == 5 );
        REQUIRE( fg->nodes() == 5 );
        REQUIRE( root->lastChild() - root->firstChild() == 1 );

        auto s0   = API::Transition(root, B[0]);
        auto s00  = API::Transition(s0,   B[0]);
        auto s000 = API::Transition(s00,  B[0]);
        auto s001 = API::Transition(s00,  B[1]);
        auto s1   = API::Transition(root, B[1]);
        auto s11  = API::Transition(s1,   B[1]);
        auto s110 = API::Transition(s11,  B[0]);
        auto s111 = API::Transition(s11,  B[1]);

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
    });
}
