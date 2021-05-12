//
// Created by tmac3 on 02/12/2020.
//

#include "catch.hpp"
#include <torch/torch.h>
#include <iostream>
#include <helpers/UnitTests.h>
#include "distributions/Dirichlet.h"
#include "math/Ops.h"

using namespace torch;
using namespace hopi::distributions;
using namespace hopi::math;
using namespace tests;

TEST_CASE( "Dirichlet distribution returns the proper type" ) {
    UnitTests::run([](){
        Dirichlet c = Dirichlet(Ops::uniformColumnWise({3}));
        REQUIRE( c.type() == DistributionType::DIRICHLET );
    });
}

TEST_CASE( "Dirichlet distribution returns the correct log params" ) {
    UnitTests::run([](){
        Tensor param1 = torch::tensor({0.1,0.7,0.2});
        Dirichlet c1 = Dirichlet(param1);
        REQUIRE( torch::equal(c1.logParams(), param1.log()) );

        Tensor param2 = Ops::uniformColumnWise({2,1});
        Dirichlet c2 = Dirichlet(param2);
        REQUIRE( torch::equal(c2.logParams(), param2.log()) );
    });
}

TEST_CASE( "Dirichlet::entropy() returns the proper results (1D)" ) {
    UnitTests::run([](){
        Dirichlet c = Dirichlet(torch::tensor({0.7,0.2,0.1}));
        REQUIRE( c.entropy() == Approx(-7.8079249494) );

        Dirichlet c1 = Dirichlet(torch::tensor({0.5,0.2,0.3}));
        REQUIRE( c1.entropy() == Approx(-1.65334191) );

        Dirichlet c2 = Dirichlet(torch::tensor({0.3,0.3,0.4}));
        REQUIRE( c2.entropy() == Approx(-0.8572948628) );
    });
}

TEST_CASE( "Dirichlet::entropy() returns the proper results (2D)" ) {
    UnitTests::run([](){
        Dirichlet c = Dirichlet(torch::tensor({0.7,1,0.2,2,0.1,3}).view({3,2}));
        REQUIRE( c.entropy() == Approx(-7.7839165065) );

        Dirichlet c1 = Dirichlet(torch::tensor({0.5,3,0.2,10,0.3,10}).view({3,2}));
        REQUIRE( c1.entropy() == Approx(-2.5557694149) );

        Dirichlet c2 = Dirichlet(torch::tensor({0.3,100,0.3,100,0.4,100}).view({3,2}));
        REQUIRE( c2.entropy() == Approx(-4.1286262513) );
    });
}

TEST_CASE( "Dirichlet::entropy() returns the proper results (3D)" ) {
    UnitTests::run([](){
        Dirichlet c0 = Dirichlet(torch::tensor({0.7,1,0.2,2,0.1,3,0.7,1,0.2,2,0.1,3}).view({2,3,2}));
        REQUIRE( c0.entropy() == Approx(-15.5678330129) );

        Dirichlet c1 = Dirichlet(torch::tensor({0.5,3,0.2,10,0.3,10,0.5,3,0.2,10,0.3,10}).view({2,3,2}));
        REQUIRE( c1.entropy() == Approx(-5.1115388297) );

        Dirichlet c2 = Dirichlet(torch::tensor({0.3,100,0.3,100,0.4,100,0.3,100,0.3,100,0.4,100}).view({2,3,2}));
        REQUIRE( c2.entropy() == Approx(-2.5557694149) );
    });
}

TEST_CASE( "Dirichlet::entropy() returns the proper results (static)" ) {
    UnitTests::run([](){
        REQUIRE( Dirichlet::entropy(torch::tensor({0.7,0.2,0.1})) == Approx(-7.8079249494) );
        REQUIRE( Dirichlet::entropy(torch::tensor({0.5,0.2,0.3})) == Approx(-1.65334191) );
        REQUIRE( Dirichlet::entropy(torch::tensor({0.3,0.3,0.4})) == Approx(-0.8572948628) );
    });
}

TEST_CASE( "Dirichlet parameters setter and getter work properly" ) {
    UnitTests::run([](){
        Tensor p1 = torch::tensor({0.2,0.8});
        Dirichlet d = Dirichlet(p1);
        REQUIRE( torch::equal(d.params(), p1) );

        Tensor p2 = torch::tensor({0.3,0.7});
        d.updateParams(p2);
        REQUIRE( torch::equal(d.params(), p2) );
    });
}

TEST_CASE( "Dirichlet parameters can be increased properly" ) {
    UnitTests::run([](){
        Tensor p1 = torch::tensor({0.2,0.8});
        Dirichlet d = Dirichlet(p1);

        REQUIRE( torch::equal(d.params(), p1) );
        d.increaseParam(0,0,0);
        REQUIRE( torch::equal(d.params(), torch::tensor({1.2,0.8})) );
        d.increaseParam(0,0,0);
        REQUIRE( torch::equal(d.params(), torch::tensor({2.2,0.8})) );
        d.increaseParam(0,1,0);
        REQUIRE( torch::equal(d.params(), torch::tensor({2.2,1.8})) );
    });
}

TEST_CASE( "Dirichlet::expectedLog() returns the proper results (1D)" ) {
    UnitTests::run([](){
        Tensor output = Dirichlet::expectedLog(torch::tensor({0.7,0.2,0.1}));
        Tensor result = torch::tensor({
            Ops::digamma(0.7) - Ops::digamma(1),
            Ops::digamma(0.2) - Ops::digamma(1),
            Ops::digamma(0.1) - Ops::digamma(1)
        });
        UnitTests::require_approximately_equal(output, result, 1);

        output = Dirichlet::expectedLog(torch::tensor({5,2,3}));
        result = torch::tensor({
            Ops::digamma(5) - Ops::digamma(10),
            Ops::digamma(2) - Ops::digamma(10),
            Ops::digamma(3) - Ops::digamma(10)
        });
        UnitTests::require_approximately_equal(output, result);

        output = Dirichlet::expectedLog(torch::tensor({30,30,40}));
        result = torch::tensor({
            Ops::digamma(30) - Ops::digamma(100),
            Ops::digamma(30) - Ops::digamma(100),
            Ops::digamma(40) - Ops::digamma(100)
        });
        UnitTests::require_approximately_equal(output, result);
    });
}

TEST_CASE( "Dirichlet::expectedLog() returns the proper results (2D)" ) {
    UnitTests::run([](){
        Tensor output = Dirichlet::expectedLog(torch::tensor({0.7,1,0.2,2,0.1,3}).view({3,2}));
        Tensor result = torch::tensor({
            Ops::digamma(0.7) - Ops::digamma(1),
            Ops::digamma(1)   - Ops::digamma(6),
            Ops::digamma(0.2) - Ops::digamma(1),
            Ops::digamma(2)   - Ops::digamma(6),
            Ops::digamma(0.1) - Ops::digamma(1),
            Ops::digamma(3)   - Ops::digamma(6)
        }).view({3,2});
        UnitTests::require_approximately_equal(output, result);

        output = Dirichlet::expectedLog(torch::tensor({5,10,2,2,3,12}).view({3,2}));
        result = torch::tensor({
            Ops::digamma(5)  - Ops::digamma(10),
            Ops::digamma(10) - Ops::digamma(24),
            Ops::digamma(2)  - Ops::digamma(10),
            Ops::digamma(2)  - Ops::digamma(24),
            Ops::digamma(3)  - Ops::digamma(10),
            Ops::digamma(12) - Ops::digamma(24)
        }).view({3,2});
        UnitTests::require_approximately_equal(output, result);

        output = Dirichlet::expectedLog(torch::tensor({30,1,30,1,40,1}).view({3,2}));
        result = torch::tensor({
            Ops::digamma(30) - Ops::digamma(100),
            Ops::digamma(1)  - Ops::digamma(3),
            Ops::digamma(30) - Ops::digamma(100),
            Ops::digamma(1)  - Ops::digamma(3),
            Ops::digamma(40) - Ops::digamma(100),
            Ops::digamma(1)  - Ops::digamma(3)
        }).view({3,2});
        UnitTests::require_approximately_equal(output, result);
    });
}

TEST_CASE( "Dirichlet::expectedLog() returns the proper results (3D)" ) {
    UnitTests::run([](){
        Tensor output = Dirichlet::expectedLog(torch::tensor({0.7,1,0.2,2,0.1,3,5,10,2,2,3,12}).view({2,3,2}));
        Tensor result = torch::tensor({
            Ops::digamma(0.7) - Ops::digamma(1),
            Ops::digamma(1)   - Ops::digamma(6),
            Ops::digamma(0.2) - Ops::digamma(1),
            Ops::digamma(2)   - Ops::digamma(6),
            Ops::digamma(0.1) - Ops::digamma(1),
            Ops::digamma(3)   - Ops::digamma(6),
            Ops::digamma(5)   - Ops::digamma(10),
            Ops::digamma(10)  - Ops::digamma(24),
            Ops::digamma(2)   - Ops::digamma(10),
            Ops::digamma(2)   - Ops::digamma(24),
            Ops::digamma(3)   - Ops::digamma(10),
            Ops::digamma(12)  - Ops::digamma(24)
        }).view({2,3,2});
        UnitTests::require_approximately_equal(output, result);
    });
}
