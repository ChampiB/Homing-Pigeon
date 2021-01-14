#include "distributions/Transition.h"
#include "distributions/ActiveTransition.h"
#include "environments/Environment.h"
#include "environments/MazeEnv.h"
#include "nodes/VarNode.h"
#include "distributions/Categorical.h"
#include "distributions/Dirichlet.h"
#include "math/Functions.h"
#include "graphs/FactorGraph.h"
#include "algorithms/AlgoTree.h"
#include "algorithms/AlgoVMP.h"
#include <Eigen/Dense>
#include <iostream>

using namespace hopi::environments;
using namespace hopi::distributions;
using namespace hopi::nodes;
using namespace hopi::graphs;
using namespace hopi::math;
using namespace hopi::algorithms;
using namespace Eigen;

int main()
{
    /**
     ** Create the environment and matrix generator.
     **/
    auto env = std::make_unique<MazeEnv>("../examples/mazes/1.maze");

    /**
     ** Create the model's parameters.
     **/
    MatrixXd theta_U = MatrixXd::Ones(env->actions(), 1);
    MatrixXd theta_A = MatrixXd::Ones(env->observations(), env->states());
    MatrixXd theta_D = MatrixXd::Ones(env->states(), 1);
    std::vector<MatrixXd> theta_B(env->actions());
    for (int i = 0; i < env->actions(); ++i) {
        theta_B[i] = MatrixXd::Ones(env->states(), env->states());
    }

    /**
     ** Create the generative model.
     **/
    VarNode *U = Dirichlet::create(theta_U);
    VarNode *A = Dirichlet::create(theta_A);
    VarNode *B = Dirichlet::create(theta_B);
    VarNode *D = Dirichlet::create(theta_D);
    VarNode *a0 = Categorical::create(U);
    VarNode *s0 = Categorical::create(D);
    VarNode *o0 = Transition::create(s0, A);
    o0->setType(VarNodeType::OBSERVED);
    o0->setName("o0");
    VarNode *s1 = ActiveTransition::create(s0, a0, B);
    VarNode *o1 = Transition::create(s1, A);
    o1->setName("o1");
    o1->setType(VarNodeType::OBSERVED);
    std::shared_ptr<FactorGraph> fg = FactorGraph::current();
    fg->setTreeRoot(s1);
    fg->loadEvidence(env->observations(), "../examples/evidences/1.evi");

    /**
     ** Create the model's prior preferences.
     **/
    MatrixXd D_tilde = MatrixXd::Constant(env->states(),  1, 1.0 / env->states());
    MatrixXd E_tilde(env->observations(),  1);
    for (int i = 0; i < env->observations(); ++i) {
        E_tilde(i, 0) = (env->observations() - i);
    }
    E_tilde = Functions::softmax(E_tilde);

    /**
     ** Run the simulation.
     **/
    env->print();
    for (int i = 0; i < 20; ++i) { // Action perception cycle
        AlgoVMP::inference(fg->getNodes());
        auto algoTree = std::make_unique<AlgoTree>(env->actions(), D_tilde, E_tilde);
        for (int j = 0; j < 100; ++j) { // Planning
            VarNode *n = algoTree->nodeSelection(fg);
            algoTree->expansion(n, A, B);
            AlgoVMP::inference(algoTree->lastExpansionNodes());
            algoTree->evaluation();
            algoTree->backpropagation(n, fg->treeRoot());
        }
        int a = algoTree->actionSelection(fg->treeRoot());
        int o = env->execute(a);
        fg->integrate(a, fg->oneHot(env->observations(), o), A, B);
        env->print();
    }

    return EXIT_SUCCESS;
}