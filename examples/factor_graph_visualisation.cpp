#include "distributions/Transition.h"
#include "distributions/ActiveTransition.h"
#include "environments/Environment.h"
#include "environments/MazeEnv.h"
#include "nodes/VarNode.h"
#include "distributions/Categorical.h"
#include "graphs/FactorGraph.h"
#include "algorithms/AlgoTree.h"
#include "algorithms/AlgoVMP.h"
#include <Eigen/Dense>

using namespace hopi::environments;
using namespace hopi::distributions;
using namespace hopi::nodes;
using namespace hopi::graphs;
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
    MatrixXd U0 = MatrixXd::Constant(env->actions(), 1, 1.0 / env->actions());
    MatrixXd A  = env->A();
    std::vector<MatrixXd> B = env->B();
    MatrixXd D0 = env->D();

    /**
     ** Create the generative model.
     **/
    VarNode *a0 = Categorical::create(U0);
    a0->setName("a0");
    VarNode *s0 = Categorical::create(D0);
    s0->setName("s0");
    VarNode *o0 = Transition::create(s0, A);
    o0->setName("o0");
    o0->setType(VarNodeType::OBSERVED);
    VarNode *s1 = ActiveTransition::create(s0, a0, B);
    s1->setName("s1");
    VarNode *o1 = Transition::create(s1, A);
    o1->setName("o1");
    o1->setType(VarNodeType::OBSERVED);
    VarNode *a1 = Categorical::create(U0);
    a1->setName("a1");
    VarNode *s2 = ActiveTransition::create(s1, a1, B);
    s2->setName("s2");
    VarNode *o2 = Transition::create(s2, A);
    o2->setName("o2");
    o2->setType(VarNodeType::OBSERVED);
    std::shared_ptr<FactorGraph> fg = FactorGraph::current();
    fg->setTreeRoot(s1);
    fg->loadEvidence(env->observations(), "../examples/evidences/1.evi");

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
    std::vector<VarNodeAttr> attrs{VarNodeAttr::G,VarNodeAttr::A,VarNodeAttr::N};
    fg->writeGraphviz("../examples/graphs/1.graph", attrs);

    return EXIT_SUCCESS;
}
