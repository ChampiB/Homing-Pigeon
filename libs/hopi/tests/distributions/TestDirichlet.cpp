//
// Created by Theophile Champion on 02/12/2020.
//

#include "catch.hpp"
#include <torch/torch.h>
#include <iostream>
#include <helpers/UnitTests.h>
#include "distributions/Dirichlet.h"
#include "math/Ops.h"
#include "api/API.h"

using namespace torch;
using namespace hopi::distributions;
using namespace hopi::math;
using namespace hopi::api;
using namespace tests;

TEST_CASE( "Dirichlet distribution returns the proper type" ) {
    UnitTests::run([](){
        Dirichlet c = Dirichlet(Ops::uniform({3}));
        REQUIRE( c.type() == DistributionType::DIRICHLET );
    });
}

TEST_CASE( "Dirichlet distribution returns the correct log params" ) {
    UnitTests::run([](){
        Tensor param1 = API::tensor({0.1,0.7,0.2});
        Dirichlet c1 = Dirichlet(param1);
        REQUIRE( equal(c1.logParams(), param1.log()) );

        Tensor param2 = Ops::uniform({2,2});
        Dirichlet c2 = Dirichlet(param2);
        REQUIRE( equal(c2.logParams(), param2.log()) );
    });
}

TEST_CASE( "Dirichlet::entropy() returns the proper results (1D)" ) {
    UnitTests::run([](){
        Dirichlet c = Dirichlet(API::tensor({0.7,0.2,0.1}));
        REQUIRE( c.entropy() == Approx(-8.7865433793) );

        Dirichlet c1 = Dirichlet(API::tensor({0.5,0.2,0.3}));
        REQUIRE( c1.entropy() == Approx(-3.3180957958) );

        Dirichlet c2 = Dirichlet(API::tensor({0.3,0.3,0.4}));
        REQUIRE( c2.entropy() == Approx(-2.2976595007) );
    });
}

TEST_CASE( "Dirichlet::entropy() returns the proper results (2D)" ) {
    UnitTests::run([](){
        Dirichlet c = Dirichlet(API::tensor({0.7,0.2,0.1,1.0,2.0,3.0}).view({2,3}));
        REQUIRE( c.entropy() == Approx(-10.0308877284) );

        Dirichlet c1 = Dirichlet(API::tensor({0.5,0.2,0.3,3.0,10.0,10.0}).view({2,3}));
        REQUIRE( c1.entropy() == Approx(-5.5976384578) );

        Dirichlet c2 = Dirichlet(API::tensor({0.3,0.3,0.4,10.0,10.0,10.0}).view({2,3}));
        REQUIRE( c2.entropy() == Approx(-4.566706217) );
    });
}

TEST_CASE( "Dirichlet::entropy() returns the proper results (3D)" ) {
    UnitTests::run([](){
        Dirichlet c0 = Dirichlet(API::tensor({0.7,1.0,0.2,2.0,0.1,3.0,0.7,1.0,0.2,2.0,0.1,3.0}).view({2,3,2}));
        REQUIRE( c0.entropy() == Approx(-22.4014435598) );

        Dirichlet c1 = Dirichlet(API::tensor({0.5,3.0,0.2,10.0,0.3,10.0,0.5,3.0,0.2,10.0,0.3,10.0}).view({2,3,2}));
        REQUIRE( c1.entropy() == Approx(-18.505914455) );

        Dirichlet c2 = Dirichlet(API::tensor({0.3,100.0,0.3,100.0,0.4,100.0,0.3,100.0,0.3,100.0,0.4,100.0}).view({2,3,2}));
        REQUIRE( c2.entropy() == Approx(-32.5352681416) );
    });
}

TEST_CASE( "Dirichlet parameters setter and getter work properly" ) {
    UnitTests::run([](){
        Tensor p1 = API::tensor({0.2,0.8});
        Dirichlet d = Dirichlet(p1);
        REQUIRE( equal(d.params(), p1) );

        Tensor p2 = API::tensor({0.3,0.7});
        d.updateParams(p2);
        REQUIRE( equal(d.params(), p2) );
    });
}

TEST_CASE( "Dirichlet::expectedLog() returns the proper results (1D)" ) {
    UnitTests::run([](){
        Tensor output = Dirichlet::expectedLog(API::tensor({0.7,0.2,0.1}));
        Tensor result = API::tensor({
            Ops::digamma(0.7) - Ops::digamma(1),
            Ops::digamma(0.2) - Ops::digamma(1),
            Ops::digamma(0.1) - Ops::digamma(1)
        });
        UnitTests::require_approximately_equal(output, result, 1);

        output = Dirichlet::expectedLog(API::tensor({5,2,3}));
        result = API::tensor({
            Ops::digamma(5) - Ops::digamma(10),
            Ops::digamma(2) - Ops::digamma(10),
            Ops::digamma(3) - Ops::digamma(10)
        });
        UnitTests::require_approximately_equal(output, result);

        output = Dirichlet::expectedLog(API::tensor({30,30,40}));
        result = API::tensor({
            Ops::digamma(30) - Ops::digamma(100),
            Ops::digamma(30) - Ops::digamma(100),
            Ops::digamma(40) - Ops::digamma(100)
        });
        UnitTests::require_approximately_equal(output, result);
    });
}

TEST_CASE( "Dirichlet::expectedLog() returns the proper results (2D)" ) {
    UnitTests::run([](){
        Tensor output = Dirichlet::expectedLog(API::tensor({0.7,0.2,0.1,1.0,2.0,3.0}).view({2,3}));
        Tensor result = API::tensor({
            Ops::digamma(0.7) - Ops::digamma(1),
            Ops::digamma(0.2) - Ops::digamma(1),
            Ops::digamma(0.1) - Ops::digamma(1),
            Ops::digamma(1)   - Ops::digamma(6),
            Ops::digamma(2)   - Ops::digamma(6),
            Ops::digamma(3)   - Ops::digamma(6)
        }).view({2,3});
        UnitTests::require_approximately_equal(output, result);

        output = Dirichlet::expectedLog(API::tensor({5,2,3,10,2,12}).view({2,3}));
        result = API::tensor({
            Ops::digamma(5)  - Ops::digamma(10),
            Ops::digamma(2)  - Ops::digamma(10),
            Ops::digamma(3)  - Ops::digamma(10),
            Ops::digamma(10) - Ops::digamma(24),
            Ops::digamma(2)  - Ops::digamma(24),
            Ops::digamma(12) - Ops::digamma(24)
        }).view({2,3});
        UnitTests::require_approximately_equal(output, result);

        output = Dirichlet::expectedLog(API::tensor({30,30,40,1,1,1}).view({2,3}));
        result = API::tensor({
            Ops::digamma(30) - Ops::digamma(100),
            Ops::digamma(30) - Ops::digamma(100),
            Ops::digamma(40) - Ops::digamma(100),
            Ops::digamma(1)  - Ops::digamma(3),
            Ops::digamma(1)  - Ops::digamma(3),
            Ops::digamma(1)  - Ops::digamma(3)
        }).view({2,3});
        UnitTests::require_approximately_equal(output, result);
    });
}

TEST_CASE( "Dirichlet::expectedLog() returns the proper results (3D)" ) {
    UnitTests::run([](){
        Tensor output = Dirichlet::expectedLog(API::tensor({0.7,0.2,0.1,1.0,2.0,3.0,5.0,2.0,3.0,10.0,2.0,12.0}).view({2,2,3}));
        Tensor result = API::tensor({
            Ops::digamma(0.7) - Ops::digamma(1),
            Ops::digamma(0.2) - Ops::digamma(1),
            Ops::digamma(0.1) - Ops::digamma(1),
            Ops::digamma(1)   - Ops::digamma(6),
            Ops::digamma(2)   - Ops::digamma(6),
            Ops::digamma(3)   - Ops::digamma(6),
            Ops::digamma(5)   - Ops::digamma(10),
            Ops::digamma(2)   - Ops::digamma(10),
            Ops::digamma(3)   - Ops::digamma(10),
            Ops::digamma(10)  - Ops::digamma(24),
            Ops::digamma(2)   - Ops::digamma(24),
            Ops::digamma(12)  - Ops::digamma(24)
        }).view({2,2,3});
        UnitTests::require_approximately_equal(output, result);
    });
}
