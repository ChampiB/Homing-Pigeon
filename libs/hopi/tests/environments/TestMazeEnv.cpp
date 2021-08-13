//
// Created by Theophile Champion on 02/12/2020.
//

#include <iostream>
#include "catch.hpp"
#include "environments/MazeEnv.h"
#include "helpers/Files.h"
#include "api/API.h"
#include <torch/torch.h>
#include <helpers/UnitTests.h>

using namespace hopi::environments;
using namespace hopi::api;
using namespace tests;
using namespace torch;

TEST_CASE( "EnvMaze throws exception for invalid files" ) {
    UnitTests::run([](){
        try {
            auto env = MazeEnv::create(Files::getMazePath("4.maze"));
            REQUIRE( false );
        } catch (const std::runtime_error& error) {
            // Do nothing
        }
    });
}

TEST_CASE( "EnvMaze do not throw exception for valid files" ) {
    UnitTests::run([](){
        try {
            auto env = MazeEnv::create(Files::getMazePath("1.maze"));
        } catch (const std::runtime_error& error) {
            REQUIRE( false );
        }
        try {
            auto env = MazeEnv::create(Files::getMazePath("2.maze"));
        } catch (const std::runtime_error& error) {
            REQUIRE( false );
        }
        try {
            auto env = MazeEnv::create(Files::getMazePath("2.maze"));
        } catch (const std::runtime_error& error) {
            REQUIRE( false );
        }
    });
}

TEST_CASE( "EnvMaze has 5 actions" ) {
    UnitTests::run([](){
        auto env = MazeEnv::create(Files::getMazePath("1.maze"));
        REQUIRE( env->actions() == 5 );
    });
}

TEST_CASE( "1.maze has 22 states" ) {
    UnitTests::run([](){
        auto env = MazeEnv::create(Files::getMazePath("1.maze"));
        REQUIRE( env->states() == 22 );
    });
}

TEST_CASE( "1.maze has 10 observations" ) {
    UnitTests::run([](){
        auto env = MazeEnv::create(Files::getMazePath("1.maze"));
        REQUIRE( env->observations() == 10 );
    });
}

TEST_CASE( "EnvMaze loads the correct agent's initial position" ) {
    UnitTests::run([](){
        auto env = MazeEnv::create(Files::getMazePath("1.maze"));
        auto pos = env->agentPosition();

        REQUIRE( pos.first  == 5 );
        REQUIRE( pos.second == 1 );
    });
}

TEST_CASE( "EnvMaze loads the correct exit's position" ) {
    UnitTests::run([](){
        auto env = MazeEnv::create(Files::getMazePath("1.maze"));
        auto pos = env->exitPosition();

        REQUIRE( pos.first  == 1 );
        REQUIRE( pos.second == 6 );
    });
}

TEST_CASE( "EnvMaze properly updates agent's position when executing actions" ) {
    UnitTests::run([](){
        auto env = MazeEnv::create(Files::getMazePath("1.maze"));

        env->execute(MazeEnv::Action::UP);
        auto pos = env->agentPosition();
        REQUIRE( pos.first  == 4 );
        REQUIRE( pos.second == 1 );

        env->execute(MazeEnv::Action::UP);
        pos = env->agentPosition();
        REQUIRE( pos.first  == 3 );
        REQUIRE( pos.second == 1 );

        env->execute(MazeEnv::Action::RIGHT);
        env->execute(MazeEnv::Action::RIGHT);
        env->execute(MazeEnv::Action::RIGHT);
        pos = env->agentPosition();
        REQUIRE( pos.first  == 3 );
        REQUIRE( pos.second == 4 );

        env->execute(MazeEnv::Action::DOWN);
        env->execute(MazeEnv::Action::DOWN);
        env->execute(MazeEnv::Action::DOWN);
        env->execute(MazeEnv::Action::DOWN);
        pos = env->agentPosition();
        REQUIRE( pos.first  == 5 );
        REQUIRE( pos.second == 4 );

        env->execute(MazeEnv::Action::LEFT);
        env->execute(MazeEnv::Action::LEFT);
        env->execute(MazeEnv::Action::LEFT);
        env->execute(MazeEnv::Action::LEFT);
        pos = env->agentPosition();
        REQUIRE( pos.first  == 5 );
        REQUIRE( pos.second == 1 );
    });
}

TEST_CASE( "EnvMaze returns correct observation" ) {
    UnitTests::run([](){
        auto env = MazeEnv::create(Files::getMazePath("1.maze"));

        REQUIRE( argmax(env->execute(MazeEnv::Action::UP)).item<int>()    == 8 );
        REQUIRE( argmax(env->execute(MazeEnv::Action::UP)).item<int>()    == 7 );
        REQUIRE( argmax(env->execute(MazeEnv::Action::UP)).item<int>()    == 6 );
        REQUIRE( argmax(env->execute(MazeEnv::Action::UP)).item<int>()    == 5 );
        REQUIRE( argmax(env->execute(MazeEnv::Action::RIGHT)).item<int>() == 4 );
        REQUIRE( argmax(env->execute(MazeEnv::Action::RIGHT)).item<int>() == 3 );
        REQUIRE( argmax(env->execute(MazeEnv::Action::RIGHT)).item<int>() == 2 );
        REQUIRE( argmax(env->execute(MazeEnv::Action::RIGHT)).item<int>() == 1 );
        REQUIRE( argmax(env->execute(MazeEnv::Action::RIGHT)).item<int>() == 0 );
        REQUIRE( argmax(env->execute(MazeEnv::Action::DOWN)).item<int>()  == 1 );
    });
}

TEST_CASE( "EnvMaze load the correct maze value" ) {
    UnitTests::run([](){
        auto env = MazeEnv::create(Files::getMazePath("1.maze"));

        // First row
        REQUIRE( (*env)(0, 0) == 1 );
        REQUIRE( (*env)(0, 1) == 1 );
        REQUIRE( (*env)(0, 2) == 1 );
        REQUIRE( (*env)(0, 3) == 1 );
        REQUIRE( (*env)(0, 4) == 1 );
        REQUIRE( (*env)(0, 5) == 1 );
        REQUIRE( (*env)(0, 6) == 1 );
        REQUIRE( (*env)(0, 7) == 1 );

        // Second row
        REQUIRE( (*env)(1, 0) == 1 );
        REQUIRE( (*env)(1, 1) == 0 );
        REQUIRE( (*env)(1, 2) == 0 );
        REQUIRE( (*env)(1, 3) == 0 );
        REQUIRE( (*env)(1, 4) == 0 );
        REQUIRE( (*env)(1, 5) == 0 );
        REQUIRE( (*env)(1, 6) == 0 );
        REQUIRE( (*env)(1, 7) == 1 );

        // Third row
        REQUIRE( (*env)(2, 0) == 1 );
        REQUIRE( (*env)(2, 1) == 0 );
        REQUIRE( (*env)(2, 2) == 1 );
        REQUIRE( (*env)(2, 3) == 1 );
        REQUIRE( (*env)(2, 4) == 1 );
        REQUIRE( (*env)(2, 5) == 1 );
        REQUIRE( (*env)(2, 6) == 0 );
        REQUIRE( (*env)(2, 7) == 1 );

        // Fourth row
        REQUIRE( (*env)(3, 0) == 1 );
        REQUIRE( (*env)(3, 1) == 0 );
        REQUIRE( (*env)(3, 2) == 0 );
        REQUIRE( (*env)(3, 3) == 0 );
        REQUIRE( (*env)(3, 4) == 0 );
        REQUIRE( (*env)(3, 5) == 1 );
        REQUIRE( (*env)(3, 6) == 0 );
        REQUIRE( (*env)(3, 7) == 1 );

        // Fifth row
        REQUIRE( (*env)(4, 0) == 1 );
        REQUIRE( (*env)(4, 1) == 0 );
        REQUIRE( (*env)(4, 2) == 1 );
        REQUIRE( (*env)(4, 3) == 1 );
        REQUIRE( (*env)(4, 4) == 0 );
        REQUIRE( (*env)(4, 5) == 1 );
        REQUIRE( (*env)(4, 6) == 0 );
        REQUIRE( (*env)(4, 7) == 1 );

        // Sixth row
        REQUIRE( (*env)(5, 0) == 1 );
        REQUIRE( (*env)(5, 1) == 0 );
        REQUIRE( (*env)(5, 2) == 0 );
        REQUIRE( (*env)(5, 3) == 0 );
        REQUIRE( (*env)(5, 4) == 0 );
        REQUIRE( (*env)(5, 5) == 0 );
        REQUIRE( (*env)(5, 6) == 0 );
        REQUIRE( (*env)(5, 7) == 1 );

        // Seventh row
        REQUIRE( (*env)(6, 0) == 1 );
        REQUIRE( (*env)(6, 1) == 1 );
        REQUIRE( (*env)(6, 2) == 1 );
        REQUIRE( (*env)(6, 3) == 1 );
        REQUIRE( (*env)(6, 4) == 1 );
        REQUIRE( (*env)(6, 5) == 1 );
        REQUIRE( (*env)(6, 6) == 1 );
        REQUIRE( (*env)(6, 7) == 1 );
    });
}

TEST_CASE( "EnvMaze properly generate A matrix" ) {
    UnitTests::run([](){
        auto env = MazeEnv::create(Files::getMazePath("5.maze"));
        Tensor A_true = API::tensor({0.025, 0.025, 0.9,   0.025, 0.025, 0.025, 0.025,
                                     0.025, 0.9,   0.025, 0.025, 0.025, 0.025, 0.025,
                                     0.9,   0.025, 0.025, 0.9,   0.025, 0.025, 0.9,
                                     0.025, 0.025, 0.025, 0.025, 0.025, 0.9,   0.025,
                                     0.025, 0.025, 0.025, 0.025, 0.9,   0.025, 0.025}).view({5,7});
        Tensor A = env->A();

        UnitTests::require_approximately_equal(A, A_true, 1);
    });
}

TEST_CASE( "EnvMaze properly generate B matrices" ) {
    UnitTests::run([](){
        auto env = MazeEnv::create(Files::getMazePath("5.maze"));
        Tensor B_true = API::tensor({
            0.9,  0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
            0.05, 0.9,  0.05, 0.9,  0.05, 0.05, 0.05,
            0.05, 0.05, 0.9,  0.05, 0.05, 0.05, 0.05,
            0.05, 0.05, 0.05, 0.05, 0.05, 0.9,  0.05,
            0.05, 0.05, 0.05, 0.05, 0.9,  0.05, 0.05,
            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.9,

            0.9,  0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
            0.05, 0.05, 0.9,  0.05, 0.05, 0.05, 0.05,
            0.05, 0.9,  0.05, 0.05, 0.05, 0.05, 0.05,
            0.05, 0.05, 0.05, 0.05, 0.9,  0.05, 0.05,
            0.05, 0.05, 0.05, 0.9,  0.05, 0.9,  0.05,
            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.9,

            0.9,  0.9,  0.05, 0.05, 0.05, 0.05, 0.05,
            0.05, 0.05, 0.9,  0.05, 0.05, 0.05, 0.05,
            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
            0.05, 0.05, 0.05, 0.9,  0.05, 0.05, 0.05,
            0.05, 0.05, 0.05, 0.05, 0.9,  0.9,  0.05,
            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.9,
            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,

            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
            0.9,  0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
            0.05, 0.9,  0.9,  0.05, 0.05, 0.05, 0.05,
            0.05, 0.05, 0.05, 0.9,  0.05, 0.05, 0.05,
            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
            0.05, 0.05, 0.05, 0.05, 0.9,  0.05, 0.05,
            0.05, 0.05, 0.05, 0.05, 0.05, 0.9,  0.9,

            0.9,  0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
            0.05, 0.9,  0.05, 0.05, 0.05, 0.05, 0.05,
            0.05, 0.05, 0.9,  0.05, 0.05, 0.05, 0.05,
            0.05, 0.05, 0.05, 0.9,  0.05, 0.05, 0.05,
            0.05, 0.05, 0.05, 0.05, 0.9,  0.05, 0.05,
            0.05, 0.05, 0.05, 0.05, 0.05, 0.9,  0.05,
            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.9
        }).view({5,7,7}).permute({1,2,0});
        Tensor B = env->B();

        UnitTests::require_approximately_equal(B, B_true, 1);
    });
}

TEST_CASE( "EnvMaze properly generate D matrix" ) {
    UnitTests::run([](){
        auto env = MazeEnv::create(Files::getMazePath("5.maze"));
        Tensor D = env->D();
        Tensor D_true = torch::tensor({0.05,0.05,0.05,0.05,0.05,0.05,0.9});

        UnitTests::require_approximately_equal(D, D_true, 1);
    });
}
