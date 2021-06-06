//
// Created by Theophile Champion on 10/05/2021.
//

#include "UnitTests.h"
#include "catch.hpp"
#include <iostream>

using namespace torch;

namespace tests{

    void UnitTests::run(void (*handler)()) {
        std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
        handler();
        std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    }

    void UnitTests::require_approximately_equal(const Tensor &t1, const Tensor &t2, double epsilon) {
        Tensor flat_t1 = flatten(t1);
        Tensor flat_t2 = flatten(t2);

        REQUIRE(t1.dim() == t2.dim());
        REQUIRE(t1.sizes() == t2.sizes());
        for (int i = 0; i < flat_t1.numel(); ++i) {
            REQUIRE( flat_t1[i].item<double>() == Approx(flat_t2[i].item<double>()).epsilon(epsilon) );
        }
    }

}
