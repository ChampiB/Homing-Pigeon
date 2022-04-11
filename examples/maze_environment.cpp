#include <memory>
#include <iostream>
#include "environments/MazeEnv.h"

using namespace hopi::environments;

void take_action(const std::unique_ptr<MazeEnv> &env, MazeEnv::Action action) {
    std::cout << std::endl << "Distance from exit: " << env->execute(action) << std::endl;
}

int main() {
    // Create the environment
    auto env = MazeEnv::create("../examples/mazes/1.lake");

    // Execute actions and display the environment state.
    env->print();
    take_action(env, MazeEnv::Action::UP);
    env->print();
    take_action(env, MazeEnv::Action::DOWN);
    env->print();
    take_action(env, MazeEnv::Action::RIGHT);
    env->print();
    take_action(env, MazeEnv::Action::LEFT);
    env->print();
    take_action(env, MazeEnv::Action::IDLE);
    env->print();

    return EXIT_SUCCESS;
}