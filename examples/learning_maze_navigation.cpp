#include "distributions/ActiveTransition.h"
#include "environments/MazeEnv.h"
#include "nodes/VarNode.h"
#include "nodes/FactorNode.h"
#include "math/Functions.h"
#include "graphs/FactorGraph.h"
#include "algorithms/AlgoTree.h"
#include "algorithms/AlgoVMP.h"
#include "api/API.h"
#include <Eigen/Dense>
#include <iostream>

using namespace hopi::environments;
using namespace hopi::distributions;
using namespace hopi::nodes;
using namespace hopi::graphs;
using namespace hopi::math;
using namespace hopi::api;
using namespace hopi::algorithms;
using namespace Eigen;

int main()
{
    /**
     ** The files' name.
     **/
    auto maze_file     = "../examples/mazes/5.maze";
    auto evidence_file = "../examples/evidences/5.evi";

    /**
     ** Create the environment and matrix generator.
     **/
    auto env = MazeEnv::create(maze_file);

    /**
     ** Hyper-parameter of the simulation.
     **/
    int nb_trials = 10;
    int nb_cycles = 25;
    int nb_planning_steps = 100;

    /**
     ** Create the model's parameters.
     **/
    std::vector<MatrixXd> theta_Us(nb_cycles);
    for (int i = 0; i < nb_cycles; ++i) {
        theta_Us[i] = MatrixXd::Ones(env->actions(), 1);
    }
    MatrixXd theta_A = MatrixXd::Ones(env->observations(), env->states());
    MatrixXd theta_D = MatrixXd::Ones(env->states(), 1);
    std::vector<MatrixXd> theta_B(env->actions());
    for (int i = 0; i < env->actions(); ++i) {
        theta_B[i] = MatrixXd::Ones(env->states(), env->states());
    }

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
    for (int i = 0; i < nb_trials; ++i) { // Trials
        // Reset the environment.
        std::cout << "Trial number: " << i << std::endl;
        env = MazeEnv::create(maze_file);

        // Create the generative model.
        FactorGraph::setCurrent(nullptr);
        std::vector<VarNode*> Us(nb_cycles);
        for (int ii = 0; ii < Us.size(); ++ii) {
            Us[ii] = API::Dirichlet(theta_Us[ii]);
        }
        VarNode *A  = API::Dirichlet(theta_A);
        VarNode *B  = API::Dirichlet(theta_B);
        VarNode *D  = API::Dirichlet(theta_D);
        VarNode *a0 = API::Categorical(Us[0]);
        VarNode *s0 = API::Categorical(D);
        VarNode *o0 = API::Transition(s0, A);
        o0->setType(VarNodeType::OBSERVED);
        o0->setName("o0");
        VarNode *s1 = API::ActiveTransition(s0, a0, B);
        VarNode *o1 = API::Transition(s1, A);
        o1->setType(VarNodeType::OBSERVED);
        o1->setName("o1");
        auto fg = FactorGraph::current();
        fg->setTreeRoot(s1);
        fg->loadEvidence(env->observations(), evidence_file);

        for (int j = 0; j < nb_cycles; ++j) { // Action perception cycle
            // Inference, planning, action selection, action execution and model integration.
            AlgoVMP::inference(fg->getNodes());
            auto algoTree = AlgoTree::create(env->actions(), D_tilde, E_tilde);
            for (int k = 0; k < nb_planning_steps; ++k) { // Planning
                VarNode *n = algoTree->nodeSelection(fg);
                algoTree->expansion(n, A, B);
                AlgoVMP::inference(algoTree->lastExpandedNodes());
                algoTree->evaluation();
                algoTree->backpropagation(n, fg->treeRoot());
            }
            int a = algoTree->actionSelection(fg->treeRoot());
            int o = env->execute(a);
            fg->integrate(Us[j], a, Functions::oneHot(env->observations(), o), A, B);
            env->print();
        }

        // Performs empirical prior:
        // i.e. posterior parameters become prior parameters for the next time point
        for (int ii = 0; ii < nb_cycles; ++ii) {
            theta_Us[ii] = Us[ii]->posterior()->params()[0];
        }
        theta_A = A->posterior()->params()[0];
        theta_D = D->posterior()->params()[0];
        for (int ii = 0; ii < env->actions(); ++ii) {
            theta_B[ii] = B->posterior()->params()[ii];
        }

        std::cout << "VFE: " << AlgoVMP::vfe(fg->getNodes()) << std::endl;
    }

    // Display the matrices of parameters.
    for (int i = 0; i < theta_Us.size(); ++i) {
        std::cout << "U[" << i << "]: " << theta_Us[i] << std::endl;
    }
    std::cout << "A: " << theta_A << std::endl;
    for (int i = 0; i < theta_B.size(); ++i) {
        std::cout << "B[" << i << "]: " << theta_B[i] << std::endl;
    }
    std::cout << "D: " << theta_D << std::endl;

    return EXIT_SUCCESS;
}
