//
// Created by Theophile Champion on 08/12/2020.
//

#ifndef HOMING_PIGEON_PROPAGATION_TYPE_H
#define HOMING_PIGEON_PROPAGATION_TYPE_H

namespace hopi::algorithms::planning {

    enum PropagationType : int {
        NO_PROP =         0, // No propagation,                       i.e. G_parent = G_parent
        UPWARD_PROP =     1, // Upward propagation,                   i.e. G_parent = G_parent + gamma * G_child
        DOWNWARD_PROP =   2, // Downward propagation,                 i.e. G_child  = G_parent + G_child
        MIN_UPWARD_PROP = 3  // Upward propagation of minimal values, i.e. G_parent = G_parent + gamma * min G_child
    };

}

#endif //HOMING_PIGEON_PROPAGATION_TYPE_H
