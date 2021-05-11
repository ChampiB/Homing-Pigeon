#include "distributions/Transition.h"
#include "distributions/ActiveTransition.h"
#include "environments/MazeEnv.h"
#include "nodes/VarNode.h"
#include "graphs/FactorGraph.h"
#include "api/API.h"
#include <Eigen/Dense>

using namespace hopi::environments;
using namespace hopi::distributions;
using namespace hopi::nodes;
using namespace hopi::graphs;
using namespace hopi::api;
using namespace Eigen;

int main()
{
    /**
     ** The files' name.
     **/
    std::string evidence_file = "../examples/evidences/1.evi";
    std::string maze_file     = "../examples/mazes/1.maze";
    std::string graph_file    = "../examples/graphs/1.graph";

    /**
     ** Create the environment and matrix generator.
     **/
    auto env = MazeEnv::create(maze_file);

    /**
     ** Create the model's parameters.
     **/
    MatrixXd U0 = MatrixXd::Constant(env->actions(), 1, 1.0 / env->actions());
    MatrixXd A  = env->A();
    std::vector<MatrixXd> B = env->B();
    MatrixXd D0 = env->D();

    /**
     ** Create the generative model.
     **/
    VarNode *a0 = API::Categorical(U0);
    a0->setName("a0");
    VarNode *s0 = API::Categorical(D0);
    s0->setName("s0");
    VarNode *o0 = API::Transition(s0, A);
    o0->setName("o0");
    o0->setType(VarNodeType::OBSERVED);
    VarNode *s1 = API::ActiveTransition(s0, a0, B);
    s1->setName("s1");
    VarNode *o1 = API::Transition(s1, A);
    o1->setName("o1");
    o1->setType(VarNodeType::OBSERVED);
    VarNode *a1 = API::Categorical(U0);
    a1->setName("a1");
    VarNode *s2 = API::ActiveTransition(s1, a1, B);
    s2->setName("s2");
    VarNode *o2 = API::Transition(s2, A);
    o2->setName("o2");
    o2->setType(VarNodeType::OBSERVED);
    auto fg = FactorGraph::current();
    fg->setTreeRoot(s1);
    fg->loadEvidence(env->observations(), evidence_file);

    /**
     ** Create the model's prior preferences.
     **/
    MatrixXd D_tilde = MatrixXd::Constant(env->states(),  1, 1.0 / env->states());
    MatrixXd E_tilde(env->observations(),  1);
    double sum = env->observations() * (env->observations() + 1.0) / 2.0;
    for (int i = 0; i < env->observations(); ++i) {
        E_tilde(i, 0) = (env->observations() - i) / sum;
    }

    /**
     ** Print the factor graph.
     **/
    std::vector<VarNodeAttr> attrs{VarNodeAttr::G, VarNodeAttr::A, VarNodeAttr::N};
    fg->writeGraphviz(graph_file, attrs);

    return EXIT_SUCCESS;
}
