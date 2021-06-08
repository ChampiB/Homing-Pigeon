//
// Created by Theophile Champion on 30/11/2020.
//

#ifndef HOMING_PIGEON_2_DISTRIBUTIONTYPE_H
#define HOMING_PIGEON_2_DISTRIBUTIONTYPE_H

namespace hopi::distributions {

    enum DistributionType : int {
        CATEGORICAL,       // P(x)     = Cat(param)
        TRANSITION,        // P(x|y)   = Cat(param)
        ACTIVE_TRANSITION, // P(x|y,z) = Cat(param)
        DIRICHLET          // P(x)     = Dir(param)
    };

}

#endif //HOMING_PIGEON_2_DISTRIBUTIONTYPE_H
