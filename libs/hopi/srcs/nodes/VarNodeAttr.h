//
// Created by Theophile Champion on 07/12/2020.
//

#ifndef HOMING_PIGEON_2_VAR_NODE_ATTR_H
#define HOMING_PIGEON_2_VAR_NODE_ATTR_H

namespace hopi::nodes {

    enum VarNodeAttr : int {
        N = 0, // Number of children expanded
        G = 1, // Node's cost / inverse quality
        A = 2  // Action that led to this state
    };

    static const std::string attrNames[] = { "N", "G", "U" };

}

#endif //HOMING_PIGEON_2_VAR_NODE_ATTR_H
