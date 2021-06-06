//
// Created by Theophile Champion on 08/12/2020.
//

#ifndef HOMING_PIGEON_2_BACKPROPAGATIONTYPE_H
#define HOMING_PIGEON_2_BACKPROPAGATIONTYPE_H

enum BackPropagationType : int {
    NO_BP =       0, // No back-propagation,       i.e. G_parent = G_parent
    UPWARD_BP =   1, // Upward back-propagation,   i.e. G_parent = G_parent + G_child
    DOWNWARD_BP = 2  // Downward back-propagation, i.e. G_child  = G_parent + G_child
};

#endif //HOMING_PIGEON_2_BACKPROPAGATIONTYPE_H
