//
// Created by Theophile Champion on 10/05/2021.
//

#ifndef HOMING_PIGEON_UNITTESTS_H
#define HOMING_PIGEON_UNITTESTS_H

#include <torch/torch.h>

namespace tests {

    class UnitTests {
    public:
        static void run(void (*handler)());
        static void require_approximately_equal(
            const torch::Tensor &t1, const torch::Tensor &t2,
            double epsilon = std::numeric_limits<float>::epsilon() * 100
        );
    };

}

#endif //HOMING_PIGEON_UNITTESTS_H
