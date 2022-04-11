//
// Created by Theophile Champion on 07/12/2020.
//

#ifndef HOMING_PIGEON_VAR_NODE_ATTR_H
#define HOMING_PIGEON_VAR_NODE_ATTR_H

namespace hopi::nodes {

    enum VarNodeAttr : int {
        N = 0,         // Number of children expanded
        G = 1,         // Node's cost / inverse quality
        A = 2,         // Action that led to this state
        PRUNED = 3,    // Is the node pruned?
        S = 4,         // Most likely state a posteriori
        AVG_G = 5,     // Average cost
        EXP_CONST = 6  // Exploration constant
    };

    static const std::string attrNames[] = {
            "Visits", "Cost", "Action", "Pruned",
            "State", "Avg Cost", "Exploration Term"
    };

}

#endif //HOMING_PIGEON_VAR_NODE_ATTR_H
