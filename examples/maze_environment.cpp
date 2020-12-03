#include <memory>
#include <iostream>
#include "environments/Environment.h"
#include "environments/MazeEnv.h"

using namespace hopi::environments;

void take_action(const std::shared_ptr<Environment>& env, MazeEnv::Action action) {
    std::cout << std::endl << "Distance from exit: " << env->execute(action) << std::endl;
}

int main()
{
    // Create the environment
    std::shared_ptr<Environment> env = std::make_unique<MazeEnv>("../examples/mazes/1.maze");

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