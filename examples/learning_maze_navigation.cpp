#include "distributions/Transition.h"
#include "distributions/ActiveTransition.h"
#include "environments/Environment.h"
#include "environments/MazeEnv.h"
#include "nodes/VarNode.h"
#include "nodes/FactorNode.h"
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

void applyFilter(VarNode *N, std::vector<int> &filters, int nb_states) {
    // Update the Dirichlet (prior and posterior)
    auto d = dynamic_cast<Dirichlet*>(N->prior());
    d->setFilters(filters);
    d = dynamic_cast<Dirichlet*>(N->posterior());
    d->setFilters(filters);
    // Update the Dirichlet's children (posterior)
    for (auto i = N->firstChild(); i < N->lastChild(); ++i) {
        if ((*i)->child()->type() != OBSERVED) {
            auto c = dynamic_cast<Categorical*>((*i)->child()->posterior());
            c->setFilters(nb_states);
        }
    }
}


int main()
{
    /**
     ** Create the environment and matrix generator.
     **/
    auto maze_file = "../examples/mazes/5.maze";
    auto evidence_file = "../examples/evidences/5.evi";
    auto env = std::make_unique<MazeEnv>(maze_file);

    /**
     ** Hyper-parameter of the simulation.
     **/
    int initial_nb_states = 1; // Number of hidden state at the beginning of the simulation
    int nb_iter_expand = 2;    // How many trials between each expansion of the state space.
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
     * Create Dirichlet filters progressively expanding the latent space.
     */
    std::vector<int> A_filters{1, env->observations(), initial_nb_states};
    std::vector<int> D_filters{1, initial_nb_states, 1};
    std::vector<int> B_filters{env->actions(), initial_nb_states, initial_nb_states};

    /**
     ** Run the simulation.
     **/
    env->print();
    for (int i = 0; i < nb_trials; ++i) { // Trials
        // Reset the environment.
        std::cout << "Trial number: " << i << std::endl;
        env = std::make_unique<MazeEnv>(maze_file);

        // Create the generative model.
        FactorGraph::setCurrent(nullptr);
        std::vector<VarNode*> Us(nb_cycles);
        for (int ii = 0; ii < Us.size(); ++ii) {
            Us[ii] = Dirichlet::create(theta_Us[ii]);
        }
        VarNode *A = Dirichlet::create(theta_A);
        VarNode *B = Dirichlet::create(theta_B);
        VarNode *D = Dirichlet::create(theta_D);
        VarNode *a0 = Categorical::create(Us[0]);
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
        fg->loadEvidence(env->observations(), evidence_file);

        // Update the Dirichlet filters if required.
        // Note that filters hide/mask part of the hidden states.
        // For example: D_filters = [1,2,1] means that only two hidden states are visible.
        // For example: A_filters = [1,3,2] means that only two hidden states are visible.
        // For example: B_filters = [5,2,2] means that only two hidden states are visible.
        if (i != 0 && D_filters[1] < env->states() - 1 && i % nb_iter_expand == 0) {
            ++D_filters[1];
            ++A_filters[2];
            ++B_filters[1];
            ++B_filters[2];
        }
        applyFilter(D, D_filters, D_filters[1]);
        applyFilter(A, A_filters, A_filters[2]);
        applyFilter(B, B_filters, B_filters[1]);

        for (int j = 0; j < nb_cycles; ++j) { // Action perception cycle
            // Inference, planning, action selection, action execution and model integration.
            AlgoVMP::inference(fg->getNodes());
            auto algoTree = std::make_unique<AlgoTree>(env->actions(), D_tilde, E_tilde);
            for (int k = 0; k < nb_planning_steps; ++k) { // Planning
                VarNode *n = algoTree->nodeSelection(fg);
                algoTree->expansion(n, A, B);
                AlgoVMP::inference(algoTree->lastExpandedNodes());
                algoTree->evaluation();
                algoTree->backpropagation(n, fg->treeRoot());
            }
            int a = algoTree->actionSelection(fg->treeRoot());
            int o = env->execute(a);
            fg->integrate(Us[j], a, fg->oneHot(env->observations(), o), A, B);
            env->print();
        }

        // Performs empirical prior:
        // i.e. posterior parameters become prior parameters for the next time point
        for (int ii = 0; ii < nb_cycles; ++ii) {
            theta_Us[ii] = Us[ii]->posterior()->params()[0];
        }
        theta_A.block(0,0,A_filters[1],A_filters[2]) = A->posterior()->params()[0];
        theta_D.block(0,0,D_filters[1],D_filters[2]) = D->posterior()->params()[0];
        for (int ii = 0; ii < env->actions(); ++ii) {
            theta_B[ii].block(0,0,B_filters[1],B_filters[2]) = B->posterior()->params()[ii];
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
