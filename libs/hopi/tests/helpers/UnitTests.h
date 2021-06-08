//
// Created by Theophile Champion on 10/05/2021.
//

#ifndef HOMINGPIGEON_UNITTESTS_H
#define HOMINGPIGEON_UNITTESTS_H

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

#endif //HOMINGPIGEON_UNITTESTS_H
