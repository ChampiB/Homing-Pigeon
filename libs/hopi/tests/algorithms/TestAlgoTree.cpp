//
// Created by tmac3 on 02/12/2020.
//

#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file

#include "catch.hpp"
#include "contexts/FactorGraphContexts.h"
#include "math/Functions.h"
#include "distributions/Distribution.h"
#include "distributions/Categorical.h"
#include "distributions/Transition.h"
#include "algorithms/AlgoTree.h"
#include "graphs/FactorGraph.h"
#include "nodes/FactorNode.h"
#include "nodes/VarNode.h"
#include <Eigen/Dense>

using namespace hopi::distributions;
using namespace hopi::algorithms;
using namespace hopi::nodes;
using namespace hopi::math;
using namespace tests;
using namespace Eigen;

TEST_CASE( "AlgoTree.lastExpandedNodes returns the nodes from the last expansion." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto fg = FactorGraphContexts::context2();
    auto algo = AlgoTree(3, MatrixXd::Constant(2, 1, 0.5), MatrixXd::Constant(2, 1, 0.5));
    MatrixXd A = MatrixXd::Constant(2, 2, 0.5);
    std::vector<MatrixXd> B {
            MatrixXd::Constant(2, 2, 0.5),
            MatrixXd::Constant(2, 2, 0.5),
            MatrixXd::Constant(2, 2, 0.5)
    };

    auto n1 = algo.nodeSelection(fg, MIN);
    algo.expansion(n1, A, B);
    auto vec = algo.lastExpandedNodes();
    REQUIRE( vec.size() == 2 );
    REQUIRE( vec[0] == fg->node(fg->nodes() - 2) );
    REQUIRE( vec[1] == fg->node(fg->nodes() - 1) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Evaluation (AVERAGE) compute the quality of the last hidden state expanded (posterior != biased)." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto fg = FactorGraphContexts::context2();
    MatrixXd posterior = MatrixXd::Constant(2, 1, 0.5);
    auto d1 = std::make_unique<Categorical>(posterior);
    MatrixXd state_pref(2, 1);
    state_pref << 0.3,
                  0.7;
    auto d2 = std::make_unique<Categorical>(state_pref);
    auto algo = AlgoTree(3, state_pref, MatrixXd::Constant(2, 1, 0.5));
    MatrixXd A = MatrixXd::Constant(2, 2, 0.5);
    std::vector<MatrixXd> B {
            MatrixXd::Constant(2, 2, 0.5),
            MatrixXd::Constant(2, 2, 0.5),
            MatrixXd::Constant(2, 2, 0.5)
    };
    auto kl = Functions::KL(d1.get(), d2.get());

    auto n1 = algo.nodeSelection(fg, MIN);
    algo.expansion(n1, A, B); // First expansion
    algo.evaluation(AVERAGE);
    auto n2 = fg->node(fg->nodes() - 2);
    REQUIRE( n2->g() == Approx(kl / 2).epsilon(0.1) );
    algo.expansion(n1, A, B); // Second expansion
    algo.evaluation(AVERAGE);
    auto n3 = fg->node(fg->nodes() - 2);
    REQUIRE( n3->g() == Approx(kl / 2).epsilon(0.1) );
    algo.expansion(n1, A, B); // Third expansion
    algo.evaluation(AVERAGE);
    auto n4 = fg->node(fg->nodes() - 2);
    REQUIRE( n4->g() == Approx(kl / 2).epsilon(0.1) );
    algo.expansion(n4, A, B);  // Fourth expansion
    algo.evaluation(AVERAGE);
    auto n5 = fg->node(fg->nodes() - 2);
    REQUIRE( n5->g() == Approx(kl * 2 / 3).epsilon(0.1) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Evaluation (KL) compute the quality of the last hidden state expanded (posterior == biased)." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto fg = FactorGraphContexts::context2();
    auto algo = AlgoTree(3, MatrixXd::Constant(2, 1, 0.5), MatrixXd::Constant(2, 1, 0.5));
    MatrixXd A = MatrixXd::Constant(2, 2, 0.5);
    std::vector<MatrixXd> B {
            MatrixXd::Constant(2, 2, 0.5),
            MatrixXd::Constant(2, 2, 0.5),
            MatrixXd::Constant(2, 2, 0.5)
    };

    auto n1 = algo.nodeSelection(fg, MIN);
    algo.expansion(n1, A, B);
    algo.evaluation(KL);
    auto n2 = fg->node(fg->nodes() - 2);
    REQUIRE( n2->g() == 0 );
    algo.expansion(n1, A, B);
    algo.evaluation(KL);
    auto n3 = fg->node(fg->nodes() - 2);
    REQUIRE( n3->g() == 0 );
    algo.expansion(n1, A, B);
    algo.evaluation(KL);
    auto n4 = fg->node(fg->nodes() - 2);
    REQUIRE( n4->g() == 0 );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Evaluation (KL) compute the quality of the last hidden state expanded (posterior != biased)." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto fg = FactorGraphContexts::context2();
    MatrixXd posterior = MatrixXd::Constant(2, 1, 0.5);
    auto d1 = std::make_unique<Categorical>(posterior);
    MatrixXd state_pref(2, 1);
    state_pref << 0.3,
                  0.7;
    auto d2 = std::make_unique<Categorical>(state_pref);
    auto algo = AlgoTree(3, state_pref, MatrixXd::Constant(2, 1, 0.5));
    MatrixXd A = MatrixXd::Constant(2, 2, 0.5);
    std::vector<MatrixXd> B {
            MatrixXd::Constant(2, 2, 0.5),
            MatrixXd::Constant(2, 2, 0.5),
            MatrixXd::Constant(2, 2, 0.5)
    };
    auto kl = Functions::KL(d1.get(), d2.get());

    auto n1 = algo.nodeSelection(fg, MIN);
    algo.expansion(n1, A, B); // First expansion
    algo.evaluation(KL);
    auto n2 = fg->node(fg->nodes() - 2);
    REQUIRE( n2->g() == kl );
    algo.expansion(n1, A, B); // Second expansion
    algo.evaluation(KL);
    auto n3 = fg->node(fg->nodes() - 2);
    REQUIRE( n3->g() == kl );
    algo.expansion(n1, A, B); // Third expansion
    algo.evaluation(KL);
    auto n4 = fg->node(fg->nodes() - 2);
    REQUIRE( n4->g() == kl );
    algo.expansion(n4, A, B);  // Fourth expansion
    algo.evaluation(KL);
    auto n5 = fg->node(fg->nodes() - 2);
    REQUIRE( n5->g() == kl );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Evaluation (SUM) compute the quality of the last hidden state expanded (posterior == biased)." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto fg = FactorGraphContexts::context2();
    auto algo = AlgoTree(3, MatrixXd::Constant(2, 1, 0.5), MatrixXd::Constant(2, 1, 0.5));
    MatrixXd A = MatrixXd::Constant(2, 2, 0.5);
    std::vector<MatrixXd> B {
            MatrixXd::Constant(2, 2, 0.5),
            MatrixXd::Constant(2, 2, 0.5),
            MatrixXd::Constant(2, 2, 0.5)
    };

    auto n1 = algo.nodeSelection(fg, MIN);
    algo.expansion(n1, A, B);
    algo.evaluation(SUM);
    auto n2 = fg->node(fg->nodes() - 2);
    REQUIRE( n2->g() == 0 );
    algo.expansion(n1, A, B);
    algo.evaluation(SUM);
    auto n3 = fg->node(fg->nodes() - 2);
    REQUIRE( n3->g() == 0 );
    algo.expansion(n1, A, B);
    algo.evaluation(SUM);
    auto n4 = fg->node(fg->nodes() - 2);
    REQUIRE( n4->g() == 0 );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Evaluation (SUM) compute the quality of the last hidden state expanded (posterior != biased)." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto fg = FactorGraphContexts::context2();
    MatrixXd posterior = MatrixXd::Constant(2, 1, 0.5);
    auto d1 = std::make_unique<Categorical>(posterior);
    MatrixXd state_pref(2, 1);
    state_pref << 0.3,
                  0.7;
    auto d2 = std::make_unique<Categorical>(state_pref);
    auto algo = AlgoTree(3, state_pref, MatrixXd::Constant(2, 1, 0.5));
    MatrixXd A = MatrixXd::Constant(2, 2, 0.5);
    std::vector<MatrixXd> B {
            MatrixXd::Constant(2, 2, 0.5),
            MatrixXd::Constant(2, 2, 0.5),
            MatrixXd::Constant(2, 2, 0.5)
    };
    auto kl = Functions::KL(d1.get(), d2.get());

    auto n1 = algo.nodeSelection(fg, MIN);
    algo.expansion(n1, A, B); // First expansion
    algo.evaluation(SUM);
    auto n2 = fg->node(fg->nodes() - 2);
    REQUIRE( n2->g() == kl );
    algo.expansion(n1, A, B); // Second expansion
    algo.evaluation(SUM);
    auto n3 = fg->node(fg->nodes() - 2);
    REQUIRE( n3->g() == kl );
    algo.expansion(n1, A, B); // Third expansion
    algo.evaluation(SUM);
    auto n4 = fg->node(fg->nodes() - 2);
    REQUIRE( n4->g() == kl );
    algo.expansion(n4, A, B);  // Fourth expansion
    algo.evaluation(SUM);
    auto n5 = fg->node(fg->nodes() - 2);
    REQUIRE( n5->g() == 2 * kl );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Node selection (MIN) consistently returns the node with lowest G." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto fg = FactorGraphContexts::context2();
    auto algo = AlgoTree(3, MatrixXd::Constant(2, 1, 0.5), MatrixXd::Constant(2, 1, 0.5));
    MatrixXd A = MatrixXd::Constant(2, 2, 0.5);
    std::vector<MatrixXd> B {
            MatrixXd::Constant(2, 2, 0.5),
            MatrixXd::Constant(2, 2, 0.5),
            MatrixXd::Constant(2, 2, 0.5)
    };

    auto n1 = algo.nodeSelection(fg, MIN);
    algo.expansion(n1, A, B); // First expansion
    algo.evaluation(SUM);
    auto n2 = fg->node(fg->nodes() - 2);
    REQUIRE( n1 == algo.nodeSelection(fg, MIN) );
    algo.expansion(n1, A, B); // Second expansion
    algo.evaluation(SUM);
    auto n3 = fg->node(fg->nodes() - 2);
    REQUIRE( n1 == algo.nodeSelection(fg, MIN) );
    algo.expansion(n1, A, B); // Third expansion
    algo.evaluation(SUM);
    auto n4 = fg->node(fg->nodes() - 2);
    n2->setG(2);
    n3->setG(3);
    n4->setG(4);
    REQUIRE( n2 == algo.nodeSelection(fg, MIN) );
    algo.expansion(n2, A, B);  // Forth expansion
    algo.evaluation(SUM);
    auto n5 = fg->node(fg->nodes() - 2);
    n5->setG(1);
    REQUIRE( n5 == algo.nodeSelection(fg, MIN) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Node selection (SOFTMAX_SAMPLING) returns well distributed nodes." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto fg = FactorGraphContexts::context2();
    auto algo = AlgoTree(3, MatrixXd::Constant(2, 1, 0.5), MatrixXd::Constant(2, 1, 0.5));
    MatrixXd A = MatrixXd::Constant(2, 2, 0.5);
    std::vector<MatrixXd> B {
            MatrixXd::Constant(2, 2, 0.5),
            MatrixXd::Constant(2, 2, 0.5),
            MatrixXd::Constant(2, 2, 0.5)
    };
    std::vector<VarNode*> nodes;

    // Perform three expansion (one per action)
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 20; ++i) {
            auto n1 = algo.nodeSelection(fg, SAMPLING);
            REQUIRE( n1 == fg->treeRoot() );
        }
        algo.expansion(algo.nodeSelection(fg, SAMPLING), A, B); // Three first expansions
        nodes.push_back(fg->node(fg->nodes() - 2));
    }

    // Compute the true probability distribution of node sampling
    nodes[0]->setG(-0.19);
    nodes[1]->setG(-0.01);
    nodes[2]->setG(-0.8);
    MatrixXd true_p(3,1);
    true_p(nodes[0]->action(), 0) = 0.19;
    true_p(nodes[1]->action(), 0) = 0.01;
    true_p(nodes[2]->action(), 0) = 0.8;
    true_p = Functions::softmax(true_p);

    // Compute the approximate probability distribution of node sampling
    MatrixXd probability = MatrixXd::Constant(3, 1, 0);
    int N = 10000;
    for (int i = 0; i < N; ++i) {
        auto n1 = algo.nodeSelection(fg, SOFTMAX_SAMPLING);
        ++probability(n1->action(), 0);
    }
    for (int i = 0; i < probability.size(); ++i) {
        probability(i, 0) /= N;
    }

    // Compare the approximate and true distribution
    for (int i = 0; i < probability.size(); ++i) {
        REQUIRE( probability(nodes[i]->action(), 0) == Approx(true_p(nodes[i]->action(), 0)).epsilon(0.2) );
    }
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Node selection (SAMPLING) consistently returns well distributed nodes." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto fg = FactorGraphContexts::context2();
    auto algo = AlgoTree(3, MatrixXd::Constant(2, 1, 0.5), MatrixXd::Constant(2, 1, 0.5));
    MatrixXd A = MatrixXd::Constant(2, 2, 0.5);
    std::vector<MatrixXd> B {
            MatrixXd::Constant(2, 2, 0.5),
            MatrixXd::Constant(2, 2, 0.5),
            MatrixXd::Constant(2, 2, 0.5)
    };
    std::vector<VarNode*> nodes;

    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 20; ++i) {
            auto n1 = algo.nodeSelection(fg, SAMPLING);
            REQUIRE( n1 == fg->treeRoot() );
        }
        algo.expansion(algo.nodeSelection(fg, SAMPLING), A, B); // Three first expansions
        nodes.push_back(fg->node(fg->nodes() - 2));
    }
    nodes[0]->setG(-0.19);
    nodes[1]->setG(-0.01);
    nodes[2]->setG(-0.8);
    std::vector<double> probability(3);
    int N = 10000;
    for (int i = 0; i < N; ++i) {
        auto n1 = algo.nodeSelection(fg, SAMPLING);
        ++probability[n1->action()];
    }
    for (int i = 0; i < probability.size(); ++i) {
        probability[nodes[i]->action()] /= (double)N;
        REQUIRE( probability[nodes[i]->action()] == Approx(-nodes[i]->g()).epsilon(0.2) );
    }
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Expansion add two nodes and properly connect them, i.e. future state and observation." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto fg = FactorGraphContexts::context2();
    auto algo = AlgoTree(
        3,
        MatrixXd::Constant(2, 1, 0.5),
        MatrixXd::Constant(2, 1, 0.5)
    );
    auto root = fg->treeRoot();
    MatrixXd A = MatrixXd::Constant(2, 2, 0.5);
    std::vector<MatrixXd> B {
        MatrixXd::Constant(2, 2, 0.5),
        MatrixXd::Constant(2, 2, 0.5),
        MatrixXd::Constant(2, 2, 0.5)
    };
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
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "After expansion(s) unexploredActions returns only actions not in the node's children." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto fg = FactorGraphContexts::context2();
    auto algo = AlgoTree(
            3,
            MatrixXd::Constant(2, 1, 0.5),
            MatrixXd::Constant(2, 1, 0.5)
    );
    auto root = fg->treeRoot();
    MatrixXd A = MatrixXd::Constant(2, 2, 0.5);
    MatrixXd B = MatrixXd::Constant(2, 2, 0.5);

    auto sI = Transition::create(root, B);
    sI->setAction(0);
    Transition::create(sI, A);
    auto act = algo.unexploredActions(fg->treeRoot());
    REQUIRE( act.size() == 2 );
    REQUIRE( std::find(act.begin(), act.end(), 1) != act.end() );
    REQUIRE( std::find(act.begin(), act.end(), 2) != act.end() );

    sI = Transition::create(root, B);
    sI->setAction(2);
    Transition::create(sI, A);
    act = algo.unexploredActions(fg->treeRoot());
    REQUIRE( act.size() == 1 );
    REQUIRE( std::find(act.begin(), act.end(), 1) != act.end() );

    sI = Transition::create(root, B);
    sI->setAction(1);
    Transition::create(sI, A);
    act = algo.unexploredActions(fg->treeRoot());
    REQUIRE( act.empty() );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "AlgoTree stop to expand when reaching the maximal depth (max_depth == 2)." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto fg = FactorGraphContexts::context2();
    auto algo = AlgoTree(
            3,
            MatrixXd::Constant(2, 1, 0.5),
            MatrixXd::Constant(2, 1, 0.5),
            2
    );
    auto root = fg->treeRoot();
    MatrixXd A = MatrixXd::Constant(2, 2, 0.5);
    std::vector<MatrixXd> B{
            MatrixXd::Constant(2, 2, 0.5),
            MatrixXd::Constant(2, 2, 0.5),
            MatrixXd::Constant(2, 2, 0.5)
    };

    for (int i = 0; i < 12; ++i) {
        auto n = algo.nodeSelection(fg);
        algo.expansion(n, A, B);
    }

    try {
        auto n = algo.nodeSelection(fg);
        algo.expansion(n, A, B);
        REQUIRE( false );
    } catch (std::runtime_error &e) {
        // Good, no expansion have been done.
    }
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "AlgoTree stop to expand when reaching the maximal depth (max_depth == 1)." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto fg = FactorGraphContexts::context2();
    auto algo = AlgoTree(
            3,
            MatrixXd::Constant(2, 1, 0.5),
            MatrixXd::Constant(2, 1, 0.5),
            1
    );
    auto root = fg->treeRoot();
    MatrixXd A = MatrixXd::Constant(2, 2, 0.5);
    std::vector<MatrixXd> B{
            MatrixXd::Constant(2, 2, 0.5),
            MatrixXd::Constant(2, 2, 0.5),
            MatrixXd::Constant(2, 2, 0.5)
    };

    for (int i = 0; i < 3; ++i) {
        auto n = algo.nodeSelection(fg);
        algo.expansion(n, A, B);
    }

    try {
        auto n = algo.nodeSelection(fg);
        algo.expansion(n, A, B);
        REQUIRE( false );
    } catch (std::runtime_error &e) {
        // Good, no expansion have been done.
    }
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Back-propagation increases N and G on all ancestors (UPWARD_BP)." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto fg = FactorGraphContexts::context2();
    auto algo = AlgoTree(3, MatrixXd::Constant(2, 1, 0.5), MatrixXd::Constant(2, 1, 0.5));
    auto root = fg->treeRoot();
    auto B = MatrixXd::Constant(2, 2, 0.5);
    auto c0 = Transition::create(root, B);
    auto c1 = Transition::create(root, B);
    auto c2 = Transition::create(root, B);

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

    AlgoTree::backpropagation(c0, root);
    REQUIRE( root->n() == 1 );
    REQUIRE( c0->n() == 1 );
    REQUIRE( c1->n() == 0 );
    REQUIRE( c2->n() == 0 );
    REQUIRE( root->g() == 3 );
    REQUIRE( c0->g() == 2 );
    REQUIRE( c1->g() == 3 );
    REQUIRE( c2->g() == 4 );

    AlgoTree::backpropagation(c1, root);
    REQUIRE( root->n() == 2 );
    REQUIRE( c0->n() == 1 );
    REQUIRE( c1->n() == 1 );
    REQUIRE( c2->n() == 0 );
    REQUIRE( root->g() == 6 );
    REQUIRE( c0->g() == 2 );
    REQUIRE( c1->g() == 3 );
    REQUIRE( c2->g() == 4 );

    AlgoTree::backpropagation(c2, root);
    REQUIRE( root->n() == 3 );
    REQUIRE( c0->n() == 1 );
    REQUIRE( c1->n() == 1 );
    REQUIRE( c2->n() == 1 );
    REQUIRE( root->g() == 10 );
    REQUIRE( c0->g() == 2 );
    REQUIRE( c1->g() == 3 );
    REQUIRE( c2->g() == 4 );

    AlgoTree::backpropagation(c1, root);
    REQUIRE( root->n() == 4 );
    REQUIRE( c0->n() == 1 );
    REQUIRE( c1->n() == 2 );
    REQUIRE( c2->n() == 1 );
    REQUIRE( root->g() == 13 );
    REQUIRE( c0->g() == 2 );
    REQUIRE( c1->g() == 3 );
    REQUIRE( c2->g() == 4 );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Back-propagation increases N on all ancestors (NO_BP)." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto fg = FactorGraphContexts::context2();
    auto algo = AlgoTree(3, MatrixXd::Constant(2, 1, 0.5), MatrixXd::Constant(2, 1, 0.5));
    auto root = fg->treeRoot();
    auto B = MatrixXd::Constant(2, 2, 0.5);
    auto c0 = Transition::create(root, B);
    auto c1 = Transition::create(root, B);
    auto c2 = Transition::create(root, B);

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

    AlgoTree::backpropagation(c0, root, NO_BP);
    REQUIRE( root->n() == 1 );
    REQUIRE( c0->n() == 1 );
    REQUIRE( c1->n() == 0 );
    REQUIRE( c2->n() == 0 );
    REQUIRE( root->g() == 1 );
    REQUIRE( c0->g() == 2 );
    REQUIRE( c1->g() == 3 );
    REQUIRE( c2->g() == 4 );

    AlgoTree::backpropagation(c1, root, NO_BP);
    REQUIRE( root->n() == 2 );
    REQUIRE( c0->n() == 1 );
    REQUIRE( c1->n() == 1 );
    REQUIRE( c2->n() == 0 );
    REQUIRE( root->g() == 1 );
    REQUIRE( c0->g() == 2 );
    REQUIRE( c1->g() == 3 );
    REQUIRE( c2->g() == 4 );

    AlgoTree::backpropagation(c2, root, NO_BP);
    REQUIRE( root->n() == 3 );
    REQUIRE( c0->n() == 1 );
    REQUIRE( c1->n() == 1 );
    REQUIRE( c2->n() == 1 );
    REQUIRE( root->g() == 1 );
    REQUIRE( c0->g() == 2 );
    REQUIRE( c1->g() == 3 );
    REQUIRE( c2->g() == 4 );

    AlgoTree::backpropagation(c1, root, NO_BP);
    REQUIRE( root->n() == 4 );
    REQUIRE( c0->n() == 1 );
    REQUIRE( c1->n() == 2 );
    REQUIRE( c2->n() == 1 );
    REQUIRE( root->g() == 1 );
    REQUIRE( c0->g() == 2 );
    REQUIRE( c1->g() == 3 );
    REQUIRE( c2->g() == 4 );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Back-propagation increases N on all ancestors (DOWNWARD_BP)." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto fg = FactorGraphContexts::context2();
    auto algo = AlgoTree(3, MatrixXd::Constant(2, 1, 0.5), MatrixXd::Constant(2, 1, 0.5));
    auto root = fg->treeRoot();
    auto B = MatrixXd::Constant(2, 2, 0.5);
    auto c0 = Transition::create(root, B);
    auto c1 = Transition::create(root, B);
    auto c2 = Transition::create(root, B);

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

    AlgoTree::backpropagation(c0, root, DOWNWARD_BP);
    REQUIRE( root->n() == 1 );
    REQUIRE( c0->n() == 1 );
    REQUIRE( c1->n() == 0 );
    REQUIRE( c2->n() == 0 );
    REQUIRE( root->g() == 1 );
    REQUIRE( c0->g() == 3 );
    REQUIRE( c1->g() == 3 );
    REQUIRE( c2->g() == 4 );

    AlgoTree::backpropagation(c1, root, DOWNWARD_BP);
    REQUIRE( root->n() == 2 );
    REQUIRE( c0->n() == 1 );
    REQUIRE( c1->n() == 1 );
    REQUIRE( c2->n() == 0 );
    REQUIRE( root->g() == 1 );
    REQUIRE( c0->g() == 3 );
    REQUIRE( c1->g() == 4 );
    REQUIRE( c2->g() == 4 );

    AlgoTree::backpropagation(c2, root, DOWNWARD_BP);
    REQUIRE( root->n() == 3 );
    REQUIRE( c0->n() == 1 );
    REQUIRE( c1->n() == 1 );
    REQUIRE( c2->n() == 1 );
    REQUIRE( root->g() == 1 );
    REQUIRE( c0->g() == 3 );
    REQUIRE( c1->g() == 4 );
    REQUIRE( c2->g() == 5 );

    AlgoTree::backpropagation(c1, root, DOWNWARD_BP);
    REQUIRE( root->n() == 4 );
    REQUIRE( c0->n() == 1 );
    REQUIRE( c1->n() == 2 );
    REQUIRE( c2->n() == 1 );
    REQUIRE( root->g() == 1 );
    REQUIRE( c0->g() == 3 );
    REQUIRE( c1->g() == 5 );
    REQUIRE( c2->g() == 5 );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Action selection returns the child variable with the highest N." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto fg = FactorGraphContexts::context2();
    auto algo = AlgoTree(3, MatrixXd::Constant(2, 1, 0.5), MatrixXd::Constant(2, 1, 0.5));
    auto root = fg->treeRoot();
    auto B = MatrixXd::Constant(2, 2, 0.5);
    auto c0 = Transition::create(root, B);
    c0->setAction(0);
    auto c1 = Transition::create(root, B);
    c1->setAction(0);
    auto c2 = Transition::create(root, B);
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
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Before any expansion unexploredActions on the tree's root returns all actions." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto fg = FactorGraphContexts::context2();
    auto algo = AlgoTree(3, MatrixXd::Constant(2, 1, 0.5), MatrixXd::Constant(2, 1, 0.5));
    auto act = algo.unexploredActions(fg->treeRoot());

    REQUIRE( act.size() == 3 );
    REQUIRE( std::find(act.begin(), act.end(), 0) != act.end() );
    REQUIRE( std::find(act.begin(), act.end(), 1) != act.end() );
    REQUIRE( std::find(act.begin(), act.end(), 2) != act.end() );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "First node selection returns the tree's root." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto fg = FactorGraphContexts::context2();
    auto algo = AlgoTree(3, MatrixXd::Constant(2, 1, 0.5), MatrixXd::Constant(2, 1, 0.5));

    REQUIRE( fg->treeRoot() == algo.nodeSelection(fg, MIN) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "CompareQuality correctly compares node's quality of first node in the pairs." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd param = MatrixXd::Constant(5, 1, 1.0 / 5);
    std::unique_ptr<VarNode> n1 = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    std::unique_ptr<VarNode> n2 = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    auto p1 = std::make_pair<VarNode*,VarNode*>(n1.get(), n2.get());
    auto p2 = std::make_pair<VarNode*,VarNode*>(n2.get(), n2.get());
    auto p3 = std::make_pair<VarNode*,VarNode*>(n2.get(), n1.get());
    auto p4 = std::make_pair<VarNode*,VarNode*>(n1.get(), n1.get());

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
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

// TODO what if p(X = x) = 0 for some x ?
TEST_CASE( "KL of identical distributions is zero." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd param = MatrixXd::Constant(5, 1, 1.0 / 5);
    std::unique_ptr<Distribution> d1 = std::make_unique<Categorical>(param);
    std::unique_ptr<Distribution> d2 = std::make_unique<Categorical>(param);

    REQUIRE(Functions::KL(d1.get(), d2.get()) == 0 );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "KL between non-categorical distributions is not supported." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd param = MatrixXd::Constant(5, 4, 1.0 / 5);
    std::unique_ptr<Distribution> d1 = std::make_unique<Transition>(param);
    std::unique_ptr<Distribution> d2 = std::make_unique<Transition>(param);

    try {
        Functions::KL(d1.get(), d2.get());
        REQUIRE( false );
    } catch (const std::runtime_error& error) {
        REQUIRE( true );
    }
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "KL of non identical distributions is not zero." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    // First distribution
    MatrixXd param1 = MatrixXd(2, 1);
    param1 << 0.5,
              0.5;
    std::unique_ptr<Distribution> d1 = std::make_unique<Categorical>(param1);
    // Second distribution
    MatrixXd param2 = MatrixXd(2, 1);
    param2 << 0.1,
              0.9;
    std::unique_ptr<Distribution> d2 = std::make_unique<Categorical>(param2);
    // True KL divergence
    double kl_result = 0.5 * (std::log(0.5) - std::log(0.1)) + 0.5 * (std::log(0.5) - std::log(0.9));

    REQUIRE(Functions::KL(d1.get(), d2.get()) == kl_result );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}
