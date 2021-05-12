//
// Created by tmac3 on 02/12/2020.
//

#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file

#include "catch.hpp"
#include "contexts/FactorGraphContexts.h"
#include "math/Ops.h"
#include "api/API.h"
#include "distributions/Categorical.h"
#include "distributions/Transition.h"
#include "algorithms/AlgoTree.h"
#include "graphs/FactorGraph.h"
#include "nodes/FactorNode.h"
#include "nodes/VarNode.h"
#include "helpers/UnitTests.h"
#include <torch/torch.h>

using namespace hopi::distributions;
using namespace hopi::algorithms;
using namespace hopi::nodes;
using namespace hopi::math;
using namespace hopi::api;
using namespace tests;
using namespace torch;

TEST_CASE( "AlgoTree.lastExpandedNodes returns the nodes from the last expansion." ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context2();
        auto conf = AlgoTreeConfig(3, Ops::uniformColumnWise({2}), Ops::uniformColumnWise({2}));
        conf.node_selection_type = MIN;
        auto algo = AlgoTree(conf);

        auto n1 = algo.nodeSelection(fg);
        algo.expansion(n1, Ops::uniformColumnWise({2,2}), Ops::uniformColumnWise({3,2,2}));
        auto vec = algo.lastExpandedNodes();
        REQUIRE( vec.size() == 2 );
        REQUIRE( vec[0] == fg->node(fg->nodes() - 2) );
        REQUIRE( vec[1] == fg->node(fg->nodes() - 1) );
    });
}

TEST_CASE( "Evaluation (DOUBLE_KL) compute the quality of the last hidden state expanded (posterior == biased)." ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context2();
        auto conf = AlgoTreeConfig(3, Ops::uniformColumnWise({2}), Ops::uniformColumnWise({2}));
        conf.node_selection_type = MIN;
        conf.evaluation_type = DOUBLE_KL;
        auto algo = AlgoTree(conf);
        Tensor A = Ops::uniformColumnWise({2,2});
        Tensor B = Ops::uniformColumnWise({3,2,2});

        auto n1 = algo.nodeSelection(fg);
        algo.expansion(n1, A, B);
        algo.evaluation();
        auto n2 = fg->node(fg->nodes() - 2);
        REQUIRE( n2->g() == 0 );
        algo.expansion(n1, A, B);
        algo.evaluation();
        auto n3 = fg->node(fg->nodes() - 2);
        REQUIRE( n3->g() == 0 );
        algo.expansion(n1, A, B);
        algo.evaluation();
        auto n4 = fg->node(fg->nodes() - 2);
        REQUIRE( n4->g() == 0 );
    });
}

TEST_CASE( "Evaluation (DOUBLE_KL) compute the quality of the last hidden state expanded (posterior != biased)." ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context2();
        auto d1 = Categorical::create(Ops::uniformColumnWise({2}));
        Tensor state_pref = torch::tensor({0.3,0.7});
        auto d2 = Categorical::create(state_pref);
        auto conf = AlgoTreeConfig(3, state_pref, Ops::uniformColumnWise({2}));
        conf.node_selection_type = MIN;
        conf.evaluation_type = DOUBLE_KL;
        auto algo = AlgoTree(conf);
        Tensor A = Ops::uniformColumnWise({2,2});
        Tensor B = Ops::uniformColumnWise({3,2,2});
        auto kl = Ops::KL(d1.get(), d2.get());

        auto n1 = algo.nodeSelection(fg);
        algo.expansion(n1, A, B); // First expansion
        algo.evaluation();
        auto n2 = fg->node(fg->nodes() - 2);
        REQUIRE( n2->g() == kl );
        algo.expansion(n1, A, B); // Second expansion
        algo.evaluation();
        auto n3 = fg->node(fg->nodes() - 2);
        REQUIRE( n3->g() == kl );
        algo.expansion(n1, A, B); // Third expansion
        algo.evaluation();
        auto n4 = fg->node(fg->nodes() - 2);
        REQUIRE( n4->g() == kl );
        algo.expansion(n4, A, B);  // Fourth expansion
        algo.evaluation();
        auto n5 = fg->node(fg->nodes() - 2);
        REQUIRE( n5->g() == kl );
    });
}

TEST_CASE( "Node selection (MIN) consistently returns the node with lowest G." ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context2();
        auto conf = AlgoTreeConfig(3, Ops::uniformColumnWise({2}), Ops::uniformColumnWise({2}));
        conf.node_selection_type = MIN;
        auto algo = AlgoTree(conf);
        Tensor A = Ops::uniformColumnWise({2,2});
        Tensor B = Ops::uniformColumnWise({3,2,2});

        auto n1 = algo.nodeSelection(fg);
        algo.expansion(n1, A, B); // First expansion
        algo.evaluation();
        auto n2 = fg->node(fg->nodes() - 2);
        REQUIRE( n1 == algo.nodeSelection(fg) );
        algo.expansion(n1, A, B); // Second expansion
        algo.evaluation();
        auto n3 = fg->node(fg->nodes() - 2);
        REQUIRE( n1 == algo.nodeSelection(fg) );
        algo.expansion(n1, A, B); // Third expansion
        algo.evaluation();
        auto n4 = fg->node(fg->nodes() - 2);
        n2->setG(2);
        n3->setG(3);
        n4->setG(4);
        REQUIRE( n2 == algo.nodeSelection(fg) );
        algo.expansion(n2, A, B);  // Forth expansion
        algo.evaluation();
        auto n5 = fg->node(fg->nodes() - 2);
        n5->setG(1);
        REQUIRE( n5 == algo.nodeSelection(fg) );
    });
}

TEST_CASE( "Node selection (SOFTMAX_SAMPLING) returns well distributed nodes." ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context2();
        auto conf = AlgoTreeConfig(3, Ops::uniformColumnWise({2}), Ops::uniformColumnWise({2}));
        conf.node_selection_type = SOFTMAX_SAMPLING;
        auto algo = AlgoTree(conf);
        Tensor A = Ops::uniformColumnWise({2,2});
        Tensor B = Ops::uniformColumnWise({3,2,2});
        std::vector<VarNode*> n;

        // Perform three expansion (one per action)
        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 20; ++i) {
                REQUIRE( algo.nodeSelection(fg) == fg->treeRoot() );
            }
            algo.expansion(algo.nodeSelection(fg), A, B); // Three first expansions
            n.push_back(fg->node(fg->nodes() - 2));
        }

        // Compute the true p distribution of node sampling
        n[0]->setG(-0.19);
        n[1]->setG(-0.01);
        n[2]->setG(-0.8);
        Tensor true_p = torch::empty({3});
        true_p[n[0]->action()] = 0.19;
        true_p[n[1]->action()] = 0.01;
        true_p[n[2]->action()] = 0.8;
        true_p = torch::softmax(true_p, 0);

        // Compute the approximate probability distribution of node sampling
        Tensor p = torch::zeros({3});
        int N = 10000;
        for (int i = 0; i < N; ++i) {
            p[algo.nodeSelection(fg)->action()] += 1;
        }
        p /= N;

        // Compare the approximate and true distribution
        for (int i = 0; i < p.numel(); ++i) {
            REQUIRE(p[n[i]->action()].item<double>() == Approx(true_p[n[i]->action()].item<double>()).epsilon(0.2) );
        }
    });
}

TEST_CASE( "Node selection (SAMPLING) consistently returns well distributed nodes." ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context2();
        auto conf = AlgoTreeConfig(3, Ops::uniformColumnWise({2}), Ops::uniformColumnWise({2}));
        conf.node_selection_type = SAMPLING;
        auto algo = AlgoTree(conf);
        Tensor A = Ops::uniformColumnWise({2,2});
        Tensor B = Ops::uniformColumnWise({3,2,2});
        std::vector<VarNode*> nodes;

        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 20; ++i) {
                REQUIRE( algo.nodeSelection(fg) == fg->treeRoot() );
            }
            algo.expansion(algo.nodeSelection(fg), A, B); // Three first expansions
            nodes.push_back(fg->node(fg->nodes() - 2));
        }
        nodes[0]->setG(-0.19);
        nodes[1]->setG(-0.01);
        nodes[2]->setG(-0.8);

        // Compute approximate probability distribution
        Tensor p = torch::zeros({3});
        int N = 10000;
        for (int i = 0; i < N; ++i) {
            p[algo.nodeSelection(fg)->action()] += 1;
        }
        p /= N;
        for (int i = 0; i < p.numel(); ++i) {
            REQUIRE(p[nodes[i]->action()].item<double>() == Approx(-nodes[i]->g()).epsilon(0.2) );
        }
    });
}

TEST_CASE( "Expansion add two nodes and properly connect them, i.e. future state and observation." ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context2();
        auto algo = AlgoTree(3, Ops::uniformColumnWise({2}), Ops::uniformColumnWise({2}));
        auto root = fg->treeRoot();
        Tensor A = Ops::uniformColumnWise({2,2});
        Tensor B = Ops::uniformColumnWise({3,2,2});
        int nNodes = fg->nodes();
        int nFactors = fg->factors();

        algo.expansion(root, A, B);
        REQUIRE( fg->nodes() == nNodes + 2 );
        REQUIRE( fg->factors() == nFactors + 2 );
        // Check children connectivity
        REQUIRE( root->nChildren() == 2 );
        REQUIRE( *(++root->firstChild()) == fg->factor(nFactors) );
        REQUIRE( fg->factor(nFactors)->child() == fg->node(nNodes) );
        REQUIRE( fg->node(nNodes)->nChildren() == 1 );
        REQUIRE( *fg->node(nNodes)->firstChild() == fg->factor(nFactors + 1) );
        REQUIRE( fg->factor(nFactors + 1)->child() == fg->node(nNodes + 1) );
        // Check parents connectivity
        REQUIRE( fg->node(nNodes + 1)->parent() == fg->factor(nFactors + 1) );
        REQUIRE( fg->factor(nFactors + 1)->parent(0) == fg->node(nNodes) );
        REQUIRE( fg->node(nNodes)->parent() == fg->factor(nFactors) );
        REQUIRE( fg->factor(nFactors)->parent(0) == root );
    });
}

TEST_CASE( "After expansion(s) unexploredActions returns only actions not in the node's children." ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context2();
        auto algo = AlgoTree(3, Ops::uniformColumnWise({2}), Ops::uniformColumnWise({2}));
        auto root = fg->treeRoot();
        Tensor A = Ops::uniformColumnWise({2,2});
        Tensor B = Ops::uniformColumnWise({2,2});

        auto sI = API::Transition(root, B);
        sI->setAction(0);
        API::Transition(sI, A);
        auto act = algo.unexploredActions(fg->treeRoot());
        REQUIRE( act.size() == 2 );
        REQUIRE( std::find(act.begin(), act.end(), 1) != act.end() );
        REQUIRE( std::find(act.begin(), act.end(), 2) != act.end() );

        sI = API::Transition(root, B);
        sI->setAction(2);
        API::Transition(sI, A);
        act = algo.unexploredActions(fg->treeRoot());
        REQUIRE( act.size() == 1 );
        REQUIRE( std::find(act.begin(), act.end(), 1) != act.end() );

        try {
            sI = API::Transition(root, B);
            sI->setAction(1);
            API::Transition(sI, A);
            act = algo.unexploredActions(fg->treeRoot());
            REQUIRE(false);
        } catch (std::runtime_error &e) {
            REQUIRE(true);
        }
    });
}

TEST_CASE( "AlgoTree stop to expand when reaching the maximal depth (max_depth == 2)." ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context2();
        auto conf = AlgoTreeConfig(3, Ops::uniformColumnWise({2}), Ops::uniformColumnWise({2}));
        conf.max_tree_depth = 2;
        auto algo = AlgoTree(conf);
        auto root = fg->treeRoot();
        Tensor A = Ops::uniformColumnWise({2,2});
        Tensor B = Ops::uniformColumnWise({3,2,2});

        for (int i = 0; i < 12; ++i) {
            algo.expansion(algo.nodeSelection(fg), A, B);
        }

        try {
            algo.expansion(algo.nodeSelection(fg), A, B);
            REQUIRE( false );
        } catch (std::runtime_error &e) {
            // Good, no expansion have been done.
        }
    });
}

TEST_CASE( "AlgoTree stop to expand when reaching the maximal depth (max_depth == 1)." ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context2();
        auto conf = AlgoTreeConfig(3, Ops::uniformColumnWise({2}), Ops::uniformColumnWise({2}));
        conf.max_tree_depth = 1;
        auto algo = AlgoTree(conf);
        Tensor A = Ops::uniformColumnWise({2,2});
        Tensor B = Ops::uniformColumnWise({3,2,2});

        for (int i = 0; i < 3; ++i) {
            algo.expansion(algo.nodeSelection(fg), A, B);
        }

        try {
            algo.expansion(algo.nodeSelection(fg), A, B);
            REQUIRE( false );
        } catch (std::runtime_error &e) {
            // Good, no expansion have been done.
        }
    });
}

TEST_CASE( "Back-propagation increases N and G on all ancestors (UPWARD_BP)." ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context2();
        auto algo = AlgoTree(3, Ops::uniformColumnWise({2}), Ops::uniformColumnWise({2}));
        auto root = fg->treeRoot();
        auto B = Ops::uniformColumnWise({2,2});
        auto c0 = API::Transition(root, B);
        auto c1 = API::Transition(root, B);
        auto c2 = API::Transition(root, B);

        root->setG(1);
        c0->setG(2);
        c1->setG(3);
        c2->setG(4);

        REQUIRE( root->n() == 0 );
        REQUIRE( c0->n() == 0 );
        REQUIRE( c1->n() == 0 );
        REQUIRE( c2->n() == 0 );
        REQUIRE( root->g() == 1 );
        REQUIRE( c0->g() == 2 );
        REQUIRE( c1->g() == 3 );
        REQUIRE( c2->g() == 4 );

        algo.backpropagation(c0, root);
        REQUIRE( root->n() == 1 );
        REQUIRE( c0->n() == 1 );
        REQUIRE( c1->n() == 0 );
        REQUIRE( c2->n() == 0 );
        REQUIRE( root->g() == 3 );
        REQUIRE( c0->g() == 2 );
        REQUIRE( c1->g() == 3 );
        REQUIRE( c2->g() == 4 );

        algo.backpropagation(c1, root);
        REQUIRE( root->n() == 2 );
        REQUIRE( c0->n() == 1 );
        REQUIRE( c1->n() == 1 );
        REQUIRE( c2->n() == 0 );
        REQUIRE( root->g() == 6 );
        REQUIRE( c0->g() == 2 );
        REQUIRE( c1->g() == 3 );
        REQUIRE( c2->g() == 4 );

        algo.backpropagation(c2, root);
        REQUIRE( root->n() == 3 );
        REQUIRE( c0->n() == 1 );
        REQUIRE( c1->n() == 1 );
        REQUIRE( c2->n() == 1 );
        REQUIRE( root->g() == 10 );
        REQUIRE( c0->g() == 2 );
        REQUIRE( c1->g() == 3 );
        REQUIRE( c2->g() == 4 );

        algo.backpropagation(c1, root);
        REQUIRE( root->n() == 4 );
        REQUIRE( c0->n() == 1 );
        REQUIRE( c1->n() == 2 );
        REQUIRE( c2->n() == 1 );
        REQUIRE( root->g() == 13 );
        REQUIRE( c0->g() == 2 );
        REQUIRE( c1->g() == 3 );
        REQUIRE( c2->g() == 4 );
    });
}

TEST_CASE( "Back-propagation increases N on all ancestors (NO_BP)." ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context2();
        auto conf = AlgoTreeConfig(3, Ops::uniformColumnWise({2}), Ops::uniformColumnWise({2}));
        conf.back_propagation_type = NO_BP;
        auto algo = AlgoTree(conf);
        auto root = fg->treeRoot();
        auto B = Ops::uniformColumnWise({2, 2});
        auto c0 = API::Transition(root, B);
        auto c1 = API::Transition(root, B);
        auto c2 = API::Transition(root, B);

        root->setG(1);
        c0->setG(2);
        c1->setG(3);
        c2->setG(4);

        REQUIRE( root->n() == 0 );
        REQUIRE( c0->n() == 0 );
        REQUIRE( c1->n() == 0 );
        REQUIRE( c2->n() == 0 );
        REQUIRE( root->g() == 1 );
        REQUIRE( c0->g() == 2 );
        REQUIRE( c1->g() == 3 );
        REQUIRE( c2->g() == 4 );

        algo.backpropagation(c0, root);
        REQUIRE( root->n() == 1 );
        REQUIRE( c0->n() == 1 );
        REQUIRE( c1->n() == 0 );
        REQUIRE( c2->n() == 0 );
        REQUIRE( root->g() == 1 );
        REQUIRE( c0->g() == 2 );
        REQUIRE( c1->g() == 3 );
        REQUIRE( c2->g() == 4 );

        algo.backpropagation(c1, root);
        REQUIRE( root->n() == 2 );
        REQUIRE( c0->n() == 1 );
        REQUIRE( c1->n() == 1 );
        REQUIRE( c2->n() == 0 );
        REQUIRE( root->g() == 1 );
        REQUIRE( c0->g() == 2 );
        REQUIRE( c1->g() == 3 );
        REQUIRE( c2->g() == 4 );

        algo.backpropagation(c2, root);
        REQUIRE( root->n() == 3 );
        REQUIRE( c0->n() == 1 );
        REQUIRE( c1->n() == 1 );
        REQUIRE( c2->n() == 1 );
        REQUIRE( root->g() == 1 );
        REQUIRE( c0->g() == 2 );
        REQUIRE( c1->g() == 3 );
        REQUIRE( c2->g() == 4 );

        algo.backpropagation(c1, root);
        REQUIRE( root->n() == 4 );
        REQUIRE( c0->n() == 1 );
        REQUIRE( c1->n() == 2 );
        REQUIRE( c2->n() == 1 );
        REQUIRE( root->g() == 1 );
        REQUIRE( c0->g() == 2 );
        REQUIRE( c1->g() == 3 );
        REQUIRE( c2->g() == 4 );
    });
}

TEST_CASE( "Back-propagation increases N on all ancestors (DOWNWARD_BP)." ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context2();
        auto conf = AlgoTreeConfig(3, Ops::uniformColumnWise({2}), Ops::uniformColumnWise({2}));
        conf.back_propagation_type = DOWNWARD_BP;
        auto algo = AlgoTree(conf);
        auto root = fg->treeRoot();
        auto B  = Ops::uniformColumnWise({2,2});
        auto c0 = API::Transition(root, B);
        auto c1 = API::Transition(root, B);
        auto c2 = API::Transition(root, B);

        root->setG(1);
        c0->setG(2);
        c1->setG(3);
        c2->setG(4);

        REQUIRE( root->n() == 0 );
        REQUIRE( c0->n() == 0 );
        REQUIRE( c1->n() == 0 );
        REQUIRE( c2->n() == 0 );
        REQUIRE( root->g() == 1 );
        REQUIRE( c0->g() == 2 );
        REQUIRE( c1->g() == 3 );
        REQUIRE( c2->g() == 4 );

        algo.backpropagation(c0, root);
        REQUIRE( root->n() == 1 );
        REQUIRE( c0->n() == 1 );
        REQUIRE( c1->n() == 0 );
        REQUIRE( c2->n() == 0 );
        REQUIRE( root->g() == 1 );
        REQUIRE( c0->g() == 3 );
        REQUIRE( c1->g() == 3 );
        REQUIRE( c2->g() == 4 );

        algo.backpropagation(c1, root);
        REQUIRE( root->n() == 2 );
        REQUIRE( c0->n() == 1 );
        REQUIRE( c1->n() == 1 );
        REQUIRE( c2->n() == 0 );
        REQUIRE( root->g() == 1 );
        REQUIRE( c0->g() == 3 );
        REQUIRE( c1->g() == 4 );
        REQUIRE( c2->g() == 4 );

        algo.backpropagation(c2, root);
        REQUIRE( root->n() == 3 );
        REQUIRE( c0->n() == 1 );
        REQUIRE( c1->n() == 1 );
        REQUIRE( c2->n() == 1 );
        REQUIRE( root->g() == 1 );
        REQUIRE( c0->g() == 3 );
        REQUIRE( c1->g() == 4 );
        REQUIRE( c2->g() == 5 );

        algo.backpropagation(c1, root);
        REQUIRE( root->n() == 4 );
        REQUIRE( c0->n() == 1 );
        REQUIRE( c1->n() == 2 );
        REQUIRE( c2->n() == 1 );
        REQUIRE( root->g() == 1 );
        REQUIRE( c0->g() == 3 );
        REQUIRE( c1->g() == 5 );
        REQUIRE( c2->g() == 5 );
    });
}

TEST_CASE( "Action selection returns the child variable with the highest N." ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context2();
        auto algo = AlgoTree(3, Ops::uniformColumnWise({2}), Ops::uniformColumnWise({2}));
        auto root = fg->treeRoot();
        auto B = Ops::uniformColumnWise({2,2});
        auto c0 = API::Transition(root, B);
        c0->setAction(0);
        auto c1 = API::Transition(root, B);
        c1->setAction(0);
        auto c2 = API::Transition(root, B);
        c2->setAction(0);

        c0->incrementN();
        REQUIRE(algo.actionSelection(root) == c0->action() );

        c1->incrementN();
        c1->incrementN();
        REQUIRE(algo.actionSelection(root) == c1->action() );

        c2->incrementN();
        c2->incrementN();
        c2->incrementN();
        REQUIRE(algo.actionSelection(root) == c2->action() );

        c0->incrementN();
        c0->incrementN();
        c0->incrementN();
        REQUIRE(algo.actionSelection(root) == c0->action() );
    });
}

TEST_CASE( "Before any expansion unexploredActions on the tree's root returns all actions." ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context2();
        auto algo = AlgoTree(3, Ops::uniformColumnWise({2}), Ops::uniformColumnWise({2}));
        auto act = algo.unexploredActions(fg->treeRoot());

        REQUIRE( act.size() == 3 );
        REQUIRE( std::find(act.begin(), act.end(), 0) != act.end() );
        REQUIRE( std::find(act.begin(), act.end(), 1) != act.end() );
        REQUIRE( std::find(act.begin(), act.end(), 2) != act.end() );
    });

}

TEST_CASE( "First node selection returns the tree's root." ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context2();
        auto conf = AlgoTreeConfig(3, Ops::uniformColumnWise({2}), Ops::uniformColumnWise({2}));
        conf.back_propagation_type = DOWNWARD_BP;
        auto algo = AlgoTree(conf);

        REQUIRE( fg->treeRoot() == algo.nodeSelection(fg) );
    });
}

TEST_CASE( "CompareQuality correctly compares node's quality of first node in the pairs." ) {
    UnitTests::run([](){
        auto n1 = VarNode::create(VarNodeType::HIDDEN);
        auto n2 = VarNode::create(VarNodeType::HIDDEN);
        std::pair p1 = {n1.get(), n2.get()};
        std::pair p2 = {n2.get(), n2.get()};
        std::pair p3 = {n2.get(), n1.get()};
        std::pair p4 = {n1.get(), n1.get()};

        n1->setG(1);
        n2->setG(2);
        REQUIRE( AlgoTree::CompareQuality(p1, p2) == true  );
        REQUIRE( AlgoTree::CompareQuality(p2, p1) == false );
        REQUIRE( AlgoTree::CompareQuality(p4, p3) == true  );
        REQUIRE( AlgoTree::CompareQuality(p3, p4) == false );
        REQUIRE( AlgoTree::CompareQuality(p1, p1) == false );
        REQUIRE( AlgoTree::CompareQuality(p2, p2) == false );
        REQUIRE( AlgoTree::CompareQuality(p3, p3) == false );
        REQUIRE( AlgoTree::CompareQuality(p4, p4) == false );
        REQUIRE( AlgoTree::CompareQuality(p3, p1) == false );
        REQUIRE( AlgoTree::CompareQuality(p1, p3) == true  );
    });
}

// TODO what if p(X = x) = 0 for some x ?
TEST_CASE( "KL of identical distributions is zero." ) {
    UnitTests::run([](){
        auto d1 = Categorical::create(Ops::uniformColumnWise({5}));
        auto d2 = Categorical::create(Ops::uniformColumnWise({5}));

        REQUIRE(Ops::KL(d1.get(), d2.get()) == 0 );
    });
}

TEST_CASE( "KL between non-categorical distributions is not supported." ) {
    UnitTests::run([](){
        auto d1 = Transition::create(Ops::uniformColumnWise({5,4}));
        auto d2 = Transition::create(Ops::uniformColumnWise({5,4}));

        try {
            Ops::KL(d1.get(), d2.get());
            REQUIRE( false );
        } catch (const std::runtime_error& error) {
            REQUIRE( true );
        }
    });
}

TEST_CASE( "KL of non identical distributions is not zero." ) {
    UnitTests::run([](){
        // First distribution
        auto d1 = Categorical::create(Ops::uniformColumnWise({2}));
        // Second distribution
        Tensor param2 = torch::tensor({0.1,0.9});
        auto d2 = Categorical::create(param2);
        // True KL divergence
        double kl_result = 0.5 * (std::log(0.5) - std::log(0.1)) + 0.5 * (std::log(0.5) - std::log(0.9));

        REQUIRE(Ops::KL(d1.get(), d2.get()) == kl_result );
    });
}

