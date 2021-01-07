//
// Created by tmac3 on 30/11/2020.
//

#ifndef HOMING_PIGEON_2_DISTRIBUTIONTYPE_H
#define HOMING_PIGEON_2_DISTRIBUTIONTYPE_H

namespace hopi::distributions {

    enum DistributionType : int {
        CATEGORICAL,
        TRANSITION,
        ACTIVE_TRANSITION,
        DIRICHLET
    };

}

#endif //HOMING_PIGEON_2_DISTRIBUTIONTYPE_H
