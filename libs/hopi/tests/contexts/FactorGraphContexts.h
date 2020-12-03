//
// Created by Theophile Champion on 02/12/2020.
//

#ifndef HOMING_PIGEON_FACTOR_GRAPH_CONTEXTS_H
#define HOMING_PIGEON_FACTOR_GRAPH_CONTEXTS_H

#include <memory>

namespace hopi::graphs {
    class FactorGraph;
}

namespace tests {

    class FactorGraphContexts {
    public:
        static std::shared_ptr<hopi::graphs::FactorGraph> context1();
        static std::shared_ptr<hopi::graphs::FactorGraph> context2();
        static std::shared_ptr<hopi::graphs::FactorGraph> context3();
    };

}

#endif //HOMING_PIGEON_FACTOR_GRAPH_CONTEXTS_H
