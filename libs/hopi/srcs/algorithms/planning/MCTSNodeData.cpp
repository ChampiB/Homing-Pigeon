//
// Created by Theophile Champion on 16/07/2021.
//

#include "MCTSNodeData.h"

namespace hopi::algorithms::planning {

    std::unique_ptr<MCTSNodeData> MCTSNodeData::create() {
        return std::make_unique<MCTSNodeData>();
    }

    MCTSNodeData::MCTSNodeData(int n, double g, int a, bool p) {
        visits = n;
        cost = g;
        action = a;
        pruned = p;
    }

}
