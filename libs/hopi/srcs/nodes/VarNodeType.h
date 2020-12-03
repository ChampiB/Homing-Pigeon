//
// Created by Theophile Champion on 29/11/2020.
//

#ifndef HOMING_PIGEON_VAR_NODE_TYPE_H
#define HOMING_PIGEON_VAR_NODE_TYPE_H

namespace hopi::nodes {

    enum VarNodeType : int {
        OBSERVED, // Observed variable
        HIDDEN    // Hidden variable
    };

}

#endif //HOMING_PIGEON_VAR_NODE_TYPE_H
