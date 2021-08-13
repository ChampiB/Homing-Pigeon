//
// Created by Theophile Champion on 08/12/2020.
//

#ifndef HOMING_PIGEON_NODE_SELECTION_TYPE_H
#define HOMING_PIGEON_NODE_SELECTION_TYPE_H

namespace hopi::algorithms::planning {

    enum NodeSelectionType : int {
        UCT1 = 0 // Select a node using the UCT1_Criterion criterion
    };

}

#endif //HOMING_PIGEON_NODE_SELECTION_TYPE_H
