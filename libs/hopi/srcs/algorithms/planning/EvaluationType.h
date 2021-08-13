//
// Created by Theophile Champion on 07/12/2020.
//

#ifndef HOMING_PIGEON_EVALUATION_TYPE_H
#define HOMING_PIGEON_EVALUATION_TYPE_H

namespace hopi::algorithms::planning {

    enum EvaluationType : int {
        DOUBLE_KL = 0, // KL[Q(S)||V(S)] + kl[Q(O)||V(O)]
        EFE = 1,       // kl[Q(O)||V(O)] + E_Q(S)[ H[P(O|S)] ]
        G_VALUES = 2,  // Estimate of the discounted EFE
    };

}

#endif //HOMING_PIGEON_EVALUATION_TYPE_H
