//
// Created by Theophile Champion on 02/12/2020.
//

#include "catch.hpp"
#include "math/Ops.h"
#include "distributions/Categorical.h"
#include "distributions/Dirichlet.h"
#include "api/API.h"
#include <torch/torch.h>
#include <iostream>
#include <helpers/UnitTests.h>

using namespace hopi::distributions;
using namespace hopi::nodes;
using namespace hopi::math;
using namespace hopi::api;
using namespace torch;
using namespace tests;
using namespace std;

TEST_CASE( "kl of identical (Categorical) distributions is zero." ) {
    UnitTests::run([](){
        Tensor param = Ops::uniform({5});
        auto d1 = Categorical::create(param);
        auto d2 = Categorical::create(param);

        REQUIRE(Ops::kl(d1.get(), d2.get()) == 0 );
    });
}

TEST_CASE( "kl of identical (Dirichlet) distributions is zero." ) {
    UnitTests::run([](){
        Tensor param = Ops::uniform({2,5,2});
        auto d1 = Dirichlet::create(param);
        auto d2 = Dirichlet::create(param);

        REQUIRE( Ops::kl(d1.get(), d2.get()) == 0 );
    });
}

TEST_CASE( "kl of non identical (Dirichlet) distributions is not zero." ) {
    UnitTests::run([](){
        auto d1 = Dirichlet::create(Ops::uniform({2}));
        auto d2 = Dirichlet::create(API::tensor({0.1,0.9}));

        REQUIRE( Ops::kl(d1.get(), d2.get()) == Approx(1.1743590056) );
    });
}

TEST_CASE( "Beta function output correct values" ) {
    UnitTests::run([](){
        REQUIRE( Ops::beta(API::tensor({1.5,0.2}))  == Approx(4.477609374347168810412)  );
        REQUIRE( Ops::beta(API::tensor({2.0,2.0}))  == Approx(0.1666666666666666666667) );
        REQUIRE( Ops::beta(API::tensor({0.01,3.5})) == Approx(98.34009340030244331049)  );
    });
}

TEST_CASE( "Logarithm of the beta function are correctly computed" ) {
    UnitTests::run([](){
        Tensor param1 = API::tensor({1.5,0.2});
        REQUIRE( Ops::log_beta(param1) == log(Ops::beta(param1)) );

        Tensor param2 = API::tensor({2,2});
        REQUIRE( Ops::log_beta(param2) == log(Ops::beta(param2)) );

        Tensor param3 = API::tensor({0.01,3.5});
        REQUIRE( Ops::log_beta(param3) == log(Ops::beta(param3)) );
    });
}

TEST_CASE( "Digamma function output correct values" ) {
    UnitTests::run([](){
        REQUIRE( Ops::digamma(0.3458) == Approx(-3.0103)              );
        REQUIRE( Ops::digamma(2)      == Approx(0.4227).epsilon(0.1)  );
        REQUIRE( Ops::digamma(1)      == Approx(-0.5773).epsilon(0.1) );
        REQUIRE( Ops::digamma(100)    == Approx(4.5952).epsilon(0.1)  );
    });
}

TEST_CASE( "OneHot returns correct one hot vectors" ) {
    UnitTests::run([](){
        Tensor m1 = Ops::one_hot(2, 1);
        REQUIRE( equal(m1, API::tensor({0,1})) );

        Tensor m2 = Ops::one_hot(5, 0);
        REQUIRE( equal(m2, API::tensor({1,0,0,0,0})) );

        Tensor m3 = Ops::one_hot(3, 2);
        REQUIRE( equal(m3, API::tensor({0,0,1})) );
    });
}

TEST_CASE( "Ops::uniform, basic tests with dim = 0" ) {
    UnitTests::run([](){
        Tensor output = Ops::uniform({3});
        Tensor result = API::full({3}, 0.3333333);
        UnitTests::require_approximately_equal(output, result);

        output = Ops::uniform({3,2});
        result = API::full({3,2}, 0.3333333);
        UnitTests::require_approximately_equal(output, result);

        output = Ops::uniform({5,2});
        result = API::full({5,2}, 0.2);
        UnitTests::require_approximately_equal(output, result);

        output = Ops::uniform({2,5,2});
        result = API::full({2,5,2}, 0.5);
        UnitTests::require_approximately_equal(output, result);

        output = Ops::uniform({2,3,2});
        result = API::full({2,3,2}, 0.5);
        UnitTests::require_approximately_equal(output, result);

        output = Ops::uniform({5,4,2});
        result = API::full({5,4,2}, 0.2);
        UnitTests::require_approximately_equal(output, result);
    });
}

TEST_CASE( "Ops::uniform basic tests with dim != 0" ) {
    UnitTests::run([](){
        Tensor output = Ops::uniform({3,2}, 1);
        Tensor result = API::full({3,2}, 0.5);
        UnitTests::require_approximately_equal(output, result);

        output = Ops::uniform({5,2}, 1);
        result = API::full({5,2}, 0.5);
        UnitTests::require_approximately_equal(output, result);

        output = Ops::uniform({2,5,2}, 1);
        result = API::full({2,5,2}, 0.2);
        UnitTests::require_approximately_equal(output, result);

        output = Ops::uniform({2,3,2}, 1);
        result = API::full({2,3,2}, 0.3333333);
        UnitTests::require_approximately_equal(output, result);

        output = Ops::uniform({5,4,2}, 2);
        result = API::full({5,4,2}, 0.5);
        UnitTests::require_approximately_equal(output, result);
    });
}

TEST_CASE( "kl of non identical (Categorical) distributions is not zero." ) {
    UnitTests::run([](){
        auto d1 = Categorical::create(Ops::uniform({2}));
        auto d2 = Categorical::create(API::tensor({0.1,0.9}));
        double kl_result = 0.5 * (log(0.5) - log(0.1)) + 0.5 * (log(0.5) - log(0.9));

        REQUIRE(Ops::kl(d1.get(), d2.get()) == Approx(kl_result) );
    });
}

TEST_CASE( "Ops::outer_tensor_product, basic tests." ) {
    UnitTests::run([](){
        auto t1 = API::tensor({1,2,3});
        REQUIRE( equal(Ops::outer_tensor_product({&t1}), t1) );

        auto t2 = API::tensor({1,10,100});
        auto res2 = API::tensor({1,10,100,2,20,200,3,30,300}).view({3,3});
        REQUIRE( equal(Ops::outer_tensor_product({&t1, &t2}), res2) );

        auto t3 = API::tensor({0,1});
        auto res3 = API::zeros({2,3,3});
        res3[1] = API::tensor({1,10,100,2,20,200,3,30,300}).view({3,3});
        REQUIRE( equal(Ops::outer_tensor_product({&t3, &t1, &t2}), res3) );
    });
}

TEST_CASE( "Ops::unsqueeze, basic tests." ) {
    UnitTests::run([](){
        auto t1 = API::tensor({1,2,3});
        auto res1 = API::tensor({1,2,3}).view({1,3});
        Ops::unsqueeze(1,{&t1});
        REQUIRE( equal(t1, res1) );

        auto t2 = API::tensor({1,2,3});
        auto res2 = API::tensor({1,2,3}).view({1,1,3});
        Ops::unsqueeze(2,{&t2});
        REQUIRE( equal(t2, res2) );
    });
}

TEST_CASE( "Ops::expansion, basic tests." ) {
    UnitTests::run([](){
        auto t1 = API::tensor({1,2,3});
        auto res1 = API::tensor({1,2,3,1,2,3}).view({2,3});
        REQUIRE( equal(Ops::expansion(t1,2,0), res1) );

        auto res2 = API::tensor({1,2,3,1,2,3,1,2,3,1,2,3}).view({4,3});
        REQUIRE( equal(Ops::expansion(t1,4,0), res2) );

        auto res3 = API::tensor({1,1,1,2,2,2,3,3,3}).view({3,3});
        REQUIRE( equal(Ops::expansion(t1,3,1), res3) );
    });
}

TEST_CASE( "Ops::multiplication, basic tests." ) {
    UnitTests::run([](){
        // multiplication of a 2-tensor by a 1-tensor along the first dimension
        auto t1 = API::tensor({1,1,1,2,3,4}).view({2,3});
        auto t2 = API::tensor({1,10});
        auto res1 = API::tensor({1,1,1,20,30,40}).view({2,3});

        REQUIRE( equal(Ops::multiplication(t1, t2, {0}), res1) );

        // multiplication of a 2-tensor by a 1-tensor along the second dimension
        auto t3 = API::tensor({1,10,100});
        auto res2 = API::tensor({1,10,100,2,30,400}).view({2,3});

        REQUIRE( equal(Ops::multiplication(t1, t3, {1}), res2) );

        // multiplication of a 3-tensor by a 2-tensor along the second and third dimensions
        auto t4 = API::zeros({2,2,3});
        t4[0] = t1;
        auto t5 = API::tensor({1,10,100,1,10,100}).view({2,3});
        auto res3 = API::zeros({2,2,3});
        res3[0] = API::tensor({1,10,100,2,30,400}).view({2,3});

        REQUIRE( equal(Ops::multiplication(t4, t5, {1,2}), res3) );
    });
}

TEST_CASE( "Ops::average, basic tests." ) {
    UnitTests::run([](){
        // average of a 2-tensor by a 1-tensor along the first dimension with reduction
        auto t1 = API::tensor({1,1,1,2,3,4}).view({2,3});
        auto t2 = API::tensor({1,10});
        auto res1 = API::tensor({21,31,41});
        REQUIRE( equal(Ops::average(t1, t2, {0}), res1) );

        // average of a 2-tensor by a 1-tensor along the second dimension with reduction
        auto t3 = API::tensor({1,10,100});
        auto res2 = API::tensor({111,432});

        REQUIRE( equal(Ops::average(t1, t3, {1}), res2) );

        // average of a 3-tensor by a 2-tensor along the second and third dimensions with reduction
        auto t4 = API::zeros({2,2,3});
        t4[0] = t1;
        auto t5 = API::tensor({1,10,100,1,10,100}).view({2,3});
        auto res3 = API::tensor({543,0});

        REQUIRE( equal(Ops::average(t4, t5, {1,2}), res3) );

        // average of a 2-tensor by a 1-tensor along the first dimension without reduction
        auto t1_bis = API::tensor({1,1,1,2,3,4}).view({2,3});
        auto t2_bis = API::tensor({1,10});
        auto res1_bis = API::tensor({1,1,1,20,30,40}).view({2,3});

        REQUIRE( equal(Ops::average(t1_bis, t2_bis, {0}, {0}), res1_bis) );

        // average of a 2-tensor by a 1-tensor along the second dimension without reduction
        auto t3_bis = API::tensor({1,10,100});
        auto res2_bis = API::tensor({1,10,100,2,30,400}).view({2,3});

        REQUIRE( equal(Ops::average(t1_bis, t3_bis, {1}, {1}), res2_bis) );

        // average of a 3-tensor by a 2-tensor along the second and third dimensions without reduction
        auto t4_bis = API::zeros({2,2,3});
        t4_bis[0] = t1_bis;
        auto t5_bis = API::tensor({1,10,100,1,10,100}).view({2,3});
        auto res3_bis = API::zeros({2,2,3});
        res3_bis[0] = API::tensor({1,10,100,2,30,400}).view({2,3});

        REQUIRE( equal(Ops::average(t4_bis, t5_bis, {1,2}, {1,2}), res3_bis) );
    });
}
