//
// Created by Theophile Champion on 02/12/2020.
//

#define CATCH_CONFIG_MAIN  // Tells Catch to provide a main() - only do this in one cpp file

#include "catch.hpp"
#include "contexts/FactorGraphContexts.h"
#include "math/Ops.h"
#include "api/API.h"
#include "distributions/Categorical.h"
#include "algorithms/planning/MCTS.h"
#include "algorithms/planning/MCTSConfig.h"
#include "algorithms/planning/MCTSNodeData.h"
#include "graphs/FactorGraph.h"
#include "nodes/FactorNode.h"
#include "nodes/VarNode.h"
#include "helpers/UnitTests.h"
#include <torch/torch.h>

using namespace hopi::distributions;
using namespace hopi::algorithms::planning;
using namespace hopi::nodes;
using namespace hopi::math;
using namespace hopi::api;
using namespace tests;
using namespace torch;

TEST_CASE( "Node selection consistently returns the node with lowest UCT." ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context2();
        auto conf = MCTSConfig::create(Ops::uniform({2}), Ops::uniform({2}), 10, 2, 1, 1);
        auto algo = MCTS(conf);
        int nActions = 3;
        Tensor A = Ops::uniform({2,2});
        Tensor B = Ops::uniform({2,2,nActions});

        auto n1 = algo.selectNode(fg->treeRoot(), nActions);
        REQUIRE( n1 == fg->treeRoot() );
        auto nodes = MCTS::expansion(n1, A, B); // First expansion
        nodes[0]->data()->cost = 2; nodes[0]->data()->visits = 1;
        nodes[2]->data()->cost = 3; nodes[2]->data()->visits = 1;
        nodes[4]->data()->cost = 4; nodes[4]->data()->visits = 1;
        std::cout << fg->treeRoot() << std::endl;
        REQUIRE( nodes[0] == algo.selectNode(fg->treeRoot(), nActions) );
        nodes = MCTS::expansion(nodes[0], A, B); // Second expansion
        nodes[0]->data()->cost = 2; nodes[0]->data()->visits = 1;
        nodes[2]->data()->cost = 1; nodes[2]->data()->visits = 1;
        nodes[4]->data()->cost = 3; nodes[4]->data()->visits = 1;
        REQUIRE( nodes[2] == algo.selectNode(fg->treeRoot(), nActions) );
    });
}

TEST_CASE( "First node selection returns the tree's root." ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context2();
        auto conf = MCTSConfig::create(Ops::uniform({2}), Ops::uniform({2}), 10, 2, 1, 1);
        auto algo = MCTS(conf);

        REQUIRE( fg->treeRoot() == algo.selectNode(fg->treeRoot(), 3) );
    });
}

TEST_CASE( "Evaluation (DOUBLE_KL) compute the quality of the last hidden state expanded (posterior == biased)." ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context2();
        auto conf = MCTSConfig::create(Ops::uniform({2}), Ops::uniform({2}), 10, 2, 1, 1);
        auto algo = MCTS(conf);
        int nActions = 3;
        Tensor A = Ops::uniform({2,2});
        Tensor B = Ops::uniform({2,2,nActions});

        auto n1 = algo.selectNode(fg->treeRoot(), nActions);
        auto nodes = MCTS::expansion(n1, A, B);
        algo.evaluation(nodes, A, DOUBLE_KL);
        for (int i = 0; i < nodes.size(); i += 2)
            REQUIRE( nodes[i]->data()->cost == 0 );
        nodes = MCTS::expansion(n1, A, B);
        algo.evaluation(nodes, A, DOUBLE_KL);
        for (int i = 0; i < nodes.size(); i += 2)
            REQUIRE( nodes[i]->data()->cost == 0 );
        nodes = MCTS::expansion(n1, A, B);
        algo.evaluation(nodes, A, DOUBLE_KL);
        for (int i = 0; i < nodes.size(); i += 2)
            REQUIRE( nodes[i]->data()->cost == 0 );
    });
}

TEST_CASE( "Evaluation (DOUBLE_KL) compute the quality of the last hidden state expanded (posterior != biased)." ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context2();
        auto d1 = Categorical::create(Ops::uniform({2}));
        Tensor state_pref = API::tensor({0.3,0.7});
        auto d2 = Categorical::create(torch::softmax(state_pref, 0));
        auto conf = MCTSConfig::create(Ops::uniform({2}), state_pref, 10, 2, 1, 1);
        auto algo = MCTS(conf);
        int nActions = 3;
        Tensor A = Ops::uniform({2,2});
        Tensor B = Ops::uniform({2,2,nActions});
        auto kl = Ops::kl(d1.get(), d2.get());
        std::cout << "KL div: " << kl << std::endl;
        std::cout << std::endl;

        auto n1 = algo.selectNode(fg->treeRoot(), nActions);
        auto nodes = MCTS::expansion(n1, A, B); // First expansion
        algo.evaluation(nodes, A, DOUBLE_KL);
        for (int i = 0; i < nodes.size(); i += 2) {
            std::cout << "nodes[i]->data()->cost: " << nodes[i]->data()->cost << std::endl;
            REQUIRE( nodes[i]->data()->cost == kl );
        }
        nodes = MCTS::expansion(nodes[3], A, B);  // Second expansion
        algo.evaluation(nodes, A, DOUBLE_KL);
        for (int i = 0; i < nodes.size(); i += 2) {
            REQUIRE( nodes[i]->data()->cost == kl );
        }
    });
}

TEST_CASE( "Evaluation (EFE) compute the quality of the last hidden state expanded." ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context2();
        auto d1 = Categorical::create(Ops::uniform({2}));
        Tensor obs_pref = API::tensor({0.3, 0.7});
        auto d2 = Categorical::create(torch::softmax(obs_pref, 0));
        auto conf = MCTSConfig::create(obs_pref, Ops::uniform({2}), 10, 2, 1, 1);
        auto algo = MCTS(conf);
        int nActions = 3;
        Tensor A = Ops::uniform({2,2});
        Tensor B = Ops::uniform({2,2,nActions});
        auto ambiguity = Ops::average(-A.log(), A, {0,1}, {1});
        ambiguity = Ops::average(ambiguity, d1->params(), {0});
        auto efe = Ops::kl(d1.get(), d2.get()) + ambiguity.item<double>();

        auto n1 = algo.selectNode(fg->treeRoot(), nActions);
        auto nodes = MCTS::expansion(n1, A, B); // First expansion
        algo.evaluation(nodes, A, EFE);
        for (int i = 0; i < nodes.size(); i += 2)
            REQUIRE( nodes[i]->data()->cost == efe );
        nodes = MCTS::expansion(nodes[0], A, B); // Second expansion
        algo.evaluation(nodes, A, EFE);
        for (int i = 0; i < nodes.size(); i += 2)
            REQUIRE( nodes[i]->data()->cost == efe );
        nodes = MCTS::expansion(nodes[0], A, B); // Third expansion
        algo.evaluation(nodes, A, EFE);
        for (int i = 0; i < nodes.size(); i += 2)
            REQUIRE( nodes[i]->data()->cost == efe );
        nodes = MCTS::expansion(nodes[0], A, B);  // Fourth expansion
        algo.evaluation(nodes, A, EFE);
        for (int i = 0; i < nodes.size(); i += 2)
            REQUIRE( nodes[i]->data()->cost == efe );
    });
}

TEST_CASE( "Back-propagation increases N and G on all ancestors." ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context2();
        auto conf = MCTSConfig::create(Ops::uniform({2}), Ops::uniform({2}), 10, 2, 1, 1);
        auto algo = MCTS(conf);
        auto root = fg->treeRoot();
        root->data()->cost = 1; root->data()->visits = 1;
        auto A = Ops::uniform({4,2});
        auto B = Ops::uniform({2,2,3});

        auto nodes = MCTS::expansion(root, A, B);
        auto n0 = nodes[0];
        auto n1 = nodes[2];
        auto n2 = nodes[4];
        n0->data()->cost = 3;
        n1->data()->cost = 1;
        n2->data()->cost = 2;
        REQUIRE( root->data()->visits == 1 );
        REQUIRE( root->data()->cost == 1 );
        REQUIRE( n0->data()->visits == 1 );
        REQUIRE( n0->data()->cost == 3 );
        REQUIRE( n1->data()->visits == 1 );
        REQUIRE( n1->data()->cost == 1 );
        REQUIRE( n2->data()->visits == 1 );
        REQUIRE( n2->data()->cost == 2 );

        MCTS::propagation(nodes);
        REQUIRE( root->data()->visits == 2 );
        REQUIRE( root->data()->cost == 2 );
        REQUIRE( n0->data()->visits == 1 );
        REQUIRE( n0->data()->cost == 3 );
        REQUIRE( n1->data()->visits == 1 );
        REQUIRE( n1->data()->cost == 1 );
        REQUIRE( n2->data()->visits == 1 );
        REQUIRE( n2->data()->cost == 2 );

        nodes = MCTS::expansion(n0, A, B);
        auto n00 = nodes[0];
        auto n01 = nodes[2];
        auto n02 = nodes[4];
        n00->data()->cost = 2;
        n01->data()->cost = 4;
        n02->data()->cost = 5;
        MCTS::propagation(nodes);
        REQUIRE( root->data()->visits == 3 );
        REQUIRE( root->data()->cost == 4 );
        REQUIRE( n0->data()->visits == 2 );
        REQUIRE( n1->data()->visits == 1 );
        REQUIRE( n2->data()->visits == 1 );
        REQUIRE( n0->data()->cost == 5 );
        REQUIRE( n1->data()->cost == 1 );
        REQUIRE( n2->data()->cost == 2 );
    });
}

TEST_CASE( "Action selection returns the child variable with the lowest average cost." ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context2();
        auto conf = MCTSConfig::create(Ops::uniform({2}), Ops::uniform({2}), 10, 2, 1, 1000);
        auto algo = MCTS(conf);
        auto root = fg->treeRoot();
        auto B = Ops::uniform({2,2,3});
        auto c0 = API::Transition(root, B); c0->data()->action = 0;
        c0->data()->cost = 1; c0->data()->visits = 1;
        auto c1 = API::Transition(root, B); c1->data()->action = 1;
        c1->data()->cost = 1; c1->data()->visits = 1;
        auto c2 = API::Transition(root, B); c2->data()->action = 2;
        c2->data()->cost = 1; c2->data()->visits = 1;

        c0->data()->cost -= 1;
        REQUIRE(algo.selectAction(root) == c0->data()->action );

        c1->data()->cost -= 2;
        REQUIRE(algo.selectAction(root) == c1->data()->action );

        c2->data()->cost -= 3;
        REQUIRE(algo.selectAction(root) == c2->data()->action );

        c0->data()->cost -= 3;
        REQUIRE(algo.selectAction(root) == c0->data()->action );
    });
}
