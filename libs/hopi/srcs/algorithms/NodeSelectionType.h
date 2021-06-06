//
// Created by Theophile Champion on 08/12/2020.
//

#ifndef HOMING_PIGEON_2_NODESELECTIONTYPE_H
#define HOMING_PIGEON_2_NODESELECTIONTYPE_H

enum NodeSelectionType : int {
    MIN = 0,             // Select the node with the smallest cost
    SAMPLING = 1,        // Sample a node from a discrete distribution whose weights correspond to the states' quality
    SOFTMAX_SAMPLING = 2 // Sample a node from a softmax function whose inputs correspond to the states' quality
};

#endif //HOMING_PIGEON_2_NODESELECTIONTYPE_H
