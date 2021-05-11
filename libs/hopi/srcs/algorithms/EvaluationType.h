//
// Created by tmac3 on 07/12/2020.
//

#ifndef HOMING_PIGEON_2_EVALUATIONTYPE_H
#define HOMING_PIGEON_2_EVALUATIONTYPE_H

enum EvaluationType : int {
    DOUBLE_KL = 0, // KL[Q(S)||V(S)] + KL[Q(O)||V(O)]
    EFE = 1,       // KL[Q(O)||V(O)] + E_Q(S)[ H[P(O|S)] ]
};

#endif //HOMING_PIGEON_2_EVALUATIONTYPE_H
