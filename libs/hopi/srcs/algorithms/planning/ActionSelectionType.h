//
// Created by Theophile Champion on 07/12/2020.
//

#ifndef HOMING_PIGEON_ACTION_SELECTION_TYPE_H
#define HOMING_PIGEON_ACTION_SELECTION_TYPE_H

namespace hopi::algorithms::planning {

    enum ActionSelectionType : int {
        MAX_N = 0,             // Select the node with the highest number of visits
        MIN_AVG_G = 1,         // Select the node with the smallest average cost
        MAX_N_MIN_G = 2,       // Select the node with the highest number of visits break ties on smallest cost
        SOFTMAX_SAMPLING_N = 0 // Sample the action from a softmax function of the number of visits
    };

}

#endif //HOMING_PIGEON_ACTION_SELECTION_TYPE_H
