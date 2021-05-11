//
// Created by tmac3 on 10/05/2021.
//

#include "UnitTests.h"
#include "catch.hpp"
#include <iostream>

namespace tests{

    void UnitTests::run(void (*handler)()) {
        std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
        handler();
        std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    }

}
