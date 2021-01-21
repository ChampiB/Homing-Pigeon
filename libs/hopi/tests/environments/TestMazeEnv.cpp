//
// Created by tmac3 on 02/12/2020.
//

#include <iostream>
#include "catch.hpp"
#include "environments/Environment.h"
#include "environments/MazeEnv.h"
#include "helpers/Files.h"
#include <Eigen/Dense>

using namespace hopi::environments;
using namespace tests;
using namespace Eigen;

TEST_CASE( "EnvMaze throws exception for invalid files." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    try {
        std::shared_ptr<Environment> env = std::make_unique<MazeEnv>(Files::getMazePath("4.maze"));
        REQUIRE( false );
    } catch (const std::runtime_error& error) {
        // Do nothing
    }
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "EnvMaze do not throw exception for valid files." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    try {
        std::shared_ptr<Environment> env = std::make_unique<MazeEnv>(Files::getMazePath("1.maze"));
    } catch (const std::runtime_error& error) {
        REQUIRE( false );
    }
    try {
        std::shared_ptr<Environment> env = std::make_unique<MazeEnv>(Files::getMazePath("2.maze"));
    } catch (const std::runtime_error& error) {
        REQUIRE( false );
    }
    try {
        std::shared_ptr<Environment> env = std::make_unique<MazeEnv>(Files::getMazePath("2.maze"));
    } catch (const std::runtime_error& error) {
        REQUIRE( false );
    }
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "EnvMaze has 5 actions." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    std::shared_ptr<Environment> env = std::make_unique<MazeEnv>(Files::getMazePath("1.maze"));
    REQUIRE( env->actions() == 5 );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "1.maze has 22 states." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    std::shared_ptr<Environment> env = std::make_unique<MazeEnv>(Files::getMazePath("1.maze"));
    REQUIRE( env->states() == 22 );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "1.maze has 10 observations." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    std::shared_ptr<Environment> env = std::make_unique<MazeEnv>(Files::getMazePath("1.maze"));
    REQUIRE( env->observations() == 10 );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "EnvMaze loads the correct agent's initial position." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    std::shared_ptr<MazeEnv> env = std::make_unique<MazeEnv>(Files::getMazePath("1.maze"));

    auto pos = env->agentPosition();
    REQUIRE( pos.first  == 5 );
    REQUIRE( pos.second == 1 );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "EnvMaze loads the correct exit's position." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    std::shared_ptr<MazeEnv> env = std::make_unique<MazeEnv>(Files::getMazePath("1.maze"));

    auto pos = env->exitPosition();
    REQUIRE( pos.first  == 1 );
    REQUIRE( pos.second == 6 );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "EnvMaze properly updates agent's position when executing actions." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    std::shared_ptr<MazeEnv> env = std::make_unique<MazeEnv>(Files::getMazePath("1.maze"));

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
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "EnvMaze returns correct observation." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    std::shared_ptr<MazeEnv> env = std::make_unique<MazeEnv>(Files::getMazePath("1.maze"));

    REQUIRE( env->execute(MazeEnv::Action::UP) == 8);
    REQUIRE( env->execute(MazeEnv::Action::UP) == 7);
    REQUIRE( env->execute(MazeEnv::Action::UP) == 6);
    REQUIRE( env->execute(MazeEnv::Action::UP) == 5);
    REQUIRE( env->execute(MazeEnv::Action::RIGHT) == 4);
    REQUIRE( env->execute(MazeEnv::Action::RIGHT) == 3);
    REQUIRE( env->execute(MazeEnv::Action::RIGHT) == 2);
    REQUIRE( env->execute(MazeEnv::Action::RIGHT) == 1);
    REQUIRE( env->execute(MazeEnv::Action::RIGHT) == 0);
    REQUIRE( env->execute(MazeEnv::Action::DOWN) == 1);
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "EnvMaze load the correct maze value." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    std::shared_ptr<MazeEnv> env = std::make_unique<MazeEnv>(Files::getMazePath("1.maze"));

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
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "EnvMaze properly generate A matrix." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    std::shared_ptr<MazeEnv> env = std::make_unique<MazeEnv>(Files::getMazePath("5.maze"));

    MatrixXd A_true(5, 7);
    A_true << 0.025, 0.025, 0.9,   0.025, 0.025, 0.025, 0.025,
              0.025, 0.9,   0.025, 0.025, 0.025, 0.025, 0.025,
              0.9,   0.025, 0.025, 0.9,   0.025, 0.025, 0.9,
              0.025, 0.025, 0.025, 0.025, 0.025, 0.9,   0.025,
              0.025, 0.025, 0.025, 0.025, 0.9,   0.025, 0.025;

    MatrixXd A = env->A();
    REQUIRE( A.rows() == 5 );
    REQUIRE( A.cols() == 7 );

    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            REQUIRE( A(i, j) == Approx(A_true(i, j)).epsilon(1) );
        }
    }
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "EnvMaze properly generate B matrices." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    std::shared_ptr<MazeEnv> env = std::make_unique<MazeEnv>(Files::getMazePath("5.maze"));

    MatrixXd B_up_true(7, 7);
    B_up_true << 0.9,  0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                 0.05, 0.9,  0.05, 0.9,  0.05, 0.05, 0.05,
                 0.05, 0.05, 0.9,  0.05, 0.05, 0.05, 0.05,
                 0.05, 0.05, 0.05, 0.05, 0.05, 0.9,  0.05,
                 0.05, 0.05, 0.05, 0.05, 0.9,  0.05, 0.05,
                 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.9;

    MatrixXd B_down_true(7, 7);
    B_down_true << 0.9,  0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                   0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                   0.05, 0.05, 0.9,  0.05, 0.05, 0.05, 0.05,
                   0.05, 0.9,  0.05, 0.05, 0.05, 0.05, 0.05,
                   0.05, 0.05, 0.05, 0.05, 0.9,  0.05, 0.05,
                   0.05, 0.05, 0.05, 0.9,  0.05, 0.9,  0.05,
                   0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.9;

    MatrixXd B_left_true(7, 7);
    B_left_true << 0.9,  0.9,  0.05, 0.05, 0.05, 0.05, 0.05,
                   0.05, 0.05, 0.9,  0.05, 0.05, 0.05, 0.05,
                   0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                   0.05, 0.05, 0.05, 0.9,  0.05, 0.05, 0.05,
                   0.05, 0.05, 0.05, 0.05, 0.9,  0.9,  0.05,
                   0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.9,
                   0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05;

    MatrixXd B_right_true(7, 7);
    B_right_true << 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                    0.9,  0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                    0.05, 0.9,  0.9,  0.05, 0.05, 0.05, 0.05,
                    0.05, 0.05, 0.05, 0.9,  0.05, 0.05, 0.05,
                    0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                    0.05, 0.05, 0.05, 0.05, 0.9,  0.05, 0.05,
                    0.05, 0.05, 0.05, 0.05, 0.05, 0.9,  0.9;

    MatrixXd B_idle_true(7, 7);
    B_idle_true << 0.9,  0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                   0.05, 0.9,  0.05, 0.05, 0.05, 0.05, 0.05,
                   0.05, 0.05, 0.9,  0.05, 0.05, 0.05, 0.05,
                   0.05, 0.05, 0.05, 0.9,  0.05, 0.05, 0.05,
                   0.05, 0.05, 0.05, 0.05, 0.9,  0.05, 0.05,
                   0.05, 0.05, 0.05, 0.05, 0.05, 0.9,  0.05,
                   0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.9;

    std::vector<MatrixXd> B_true{B_up_true, B_down_true, B_left_true, B_right_true, B_idle_true};
    std::vector<MatrixXd> B = env->B();

    for (auto & m : B) {
        REQUIRE(m.rows() == 7 );
        REQUIRE(m.cols() == 7 );
    }

    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 7; ++j) {
            for (int k = 0; k < 7; ++k) {
                REQUIRE( B[i](j, k) == Approx(B_true[i](j, k)).epsilon(1) );
            }
        }
    }
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "EnvMaze properly generate D matrix." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    std::shared_ptr<MazeEnv> env = std::make_unique<MazeEnv>(Files::getMazePath("5.maze"));

    MatrixXd D = env->D();
    REQUIRE( D.rows() == 7 );
    REQUIRE( D.cols() == 1 );
    REQUIRE( D(0, 0) == Approx(0.05).epsilon(1) );
    REQUIRE( D(1, 0) == Approx(0.05).epsilon(1) );
    REQUIRE( D(2, 0) == Approx(0.05).epsilon(1) );
    REQUIRE( D(3, 0) == Approx(0.05).epsilon(1) );
    REQUIRE( D(4, 0) == Approx(0.05).epsilon(1) );
    REQUIRE( D(5, 0) == Approx(0.05).epsilon(1) );
    REQUIRE( D(6, 0) == 0.9 );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}
