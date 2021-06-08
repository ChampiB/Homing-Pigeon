//
// Created by Theophile Champion on 02/12/2020.
//

#include "FactorGraphContexts.h"
#include "distributions/ActiveTransition.h"
#include "graphs/FactorGraph.h"
#include "nodes/VarNode.h"
#include "api/API.h"
#include "math/Ops.h"
#include <torch/torch.h>

using namespace torch;
using namespace hopi::nodes;
using namespace hopi::graphs;
using namespace hopi::math;
using namespace hopi::api;
using namespace hopi::distributions;

namespace tests {

    std::shared_ptr<FactorGraph> FactorGraphContexts::context1() {
        FactorGraph::setCurrent(nullptr);
        auto fg = FactorGraph::current();
        Tensor param = Ops::uniform({3});

        VarNode *c1 = API::Categorical(param);
        c1->setType(VarNodeType::OBSERVED);
        API::Categorical(param); // First hidden variable
        VarNode *c3 = API::Categorical(param);
        c3->setType(VarNodeType::OBSERVED);
        VarNode *c4 = API::Categorical(param);
        c4->setType(VarNodeType::OBSERVED);
        API::Categorical(param); // Second hidden variable
        API::Categorical(param); // Third hidden variable
        VarNode *c6 = API::Categorical(param);
        c6->setType(VarNodeType::OBSERVED);
        VarNode *c7 = API::Categorical(param);
        c7->setType(VarNodeType::OBSERVED);
        VarNode *c8 = API::Categorical(param);
        c8->setType(VarNodeType::OBSERVED);
        return fg;
    }

    std::shared_ptr<hopi::graphs::FactorGraph> FactorGraphContexts::context2() {
        FactorGraph::setCurrent(nullptr);
        auto fg = FactorGraph::current();

        /**
         ** Create the model's parameters.
         **/
        Tensor U0 = Ops::uniform({5});
        Tensor D0 = Ops::uniform({3});
        Tensor A  = Ops::uniform({9,3});
        Tensor B  = Ops::uniform({3,3,5});

        /**
         ** Create the generative model.
         **/
        VarNode *a0 = API::Categorical(U0);
        VarNode *s0 = API::Categorical(D0);
        VarNode *o0 = API::Transition(s0, A);
        o0->setType(VarNodeType::OBSERVED);
        VarNode *s1 = API::ActiveTransition(s0, a0, B);
        VarNode *o1 = API::Transition(s1, A);
        o1->setType(VarNodeType::OBSERVED);
        fg->setTreeRoot(s1);

        return fg;
    }

    std::shared_ptr<hopi::graphs::FactorGraph> FactorGraphContexts::context3() {
        FactorGraph::setCurrent(nullptr);
        auto fg = FactorGraph::current();
        Tensor param = Ops::uniform({3});

        API::Categorical(param);
        VarNode *c1 = API::Categorical(param); // First observed variable
        c1->setType(VarNodeType::OBSERVED);
        API::Categorical(param);
        API::Categorical(param);
        VarNode *c3 = API::Categorical(param); // Second observed variable
        c3->setType(VarNodeType::OBSERVED);
        VarNode *c4 = API::Categorical(param); // Third observed variable
        c4->setType(VarNodeType::OBSERVED);
        API::Categorical(param);
        API::Categorical(param);
        API::Categorical(param);
        return fg;
    }

}
