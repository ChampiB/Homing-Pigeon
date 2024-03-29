#include "distributions/Transition.h"
#include "environments/MazeEnv.h"
#include "nodes/VarNode.h"
#include "math/Ops.h"
#include "api/API.h"
#include "graphs/FactorGraph.h"
#include "algorithms/planning/MCTS.h"
#include "algorithms/inference/VMP.h"
#include <torch/torch.h>
#include <iostream>

using namespace hopi::environments;
using namespace hopi::distributions;
using namespace hopi::nodes;
using namespace hopi::math;
using namespace hopi::graphs;
using namespace hopi::algorithms::inference;
using namespace hopi::algorithms::planning;
using namespace hopi::api;
using namespace torch;

int main() {
    /**
     ** Create the environment and matrix generator.
     **/
    auto env = MazeEnv::create("../examples/mazes/1.lake");

    /**
     ** Create the model's parameters.
     **/
    Tensor U0 = Ops::uniform({env->actions()});
    Tensor A  = env->A();
    Tensor B  = env->B();
    Tensor D0 = env->D();

    /**
     ** Create the generative model.
     **/
    VarNode *a0 = API::Categorical(U0);
    VarNode *s0 = API::Categorical(D0);
    VarNode *o0 = API::Transition(s0, A);
    o0->setType(VarNodeType::OBSERVED);
    o0->setName("o0");
    VarNode *s1 = API::ActiveTransition(s0, a0, B);
    VarNode *o1 = API::Transition(s1, A);
    o1->setName("o1");
    o1->setType(VarNodeType::OBSERVED);
    auto fg = FactorGraph::current();
    fg->setTreeRoot(s1);
    fg->loadEvidence(env->observations(), "../examples/evidences/1.evi");

    /**
     ** Create the model's prior preferences.
     **/
    Tensor D_tilde = Ops::uniform({env->states()});
    Tensor E_tilde = softmax(env->observations() - API::range(0, env->observations()), 0);

    /**
     ** Run the simulation.
     **/
    env->print();
    for (int i = 0; i < 20; ++i) { // Action perception cycle
        VMP::inference(fg->getNodes());
        auto algoTree = MCTS::create(env->actions(), D_tilde, E_tilde);
        for (int j = 0; j < 100; ++j) { // Planning
            VarNode *n = algoTree->nodeSelection(fg);
            algoTree->expansion(n, A, B);
            VMP::inference(algoTree->lastExpandedNodes());
            algoTree->evaluation(fg, A, B);
            algoTree->propagation(fg->treeRoot());
        }
        int a = algoTree->actionSelection(fg->treeRoot());
        int o = env->execute(a);
        fg->integrate(a, Ops::one_hot(env->observations(), o), A, B);
        env->print();
    }

    return EXIT_SUCCESS;
}
