//
// Created by Theophile Champion on 28/11/2020.
//

#ifndef HOMING_PIGEON_DISENTANGLE_SPRITES_ENV_H
#define HOMING_PIGEON_DISENTANGLE_SPRITES_ENV_H

#include "Environment.h"
#include <string>
#include <memory>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

namespace hopi::environments {

    /**
     * Class representing a environment where an agent evolve in a lake.
     */
    class DisentangleSpritesEnv : public Environment {
    public:
        //
        // Factory
        //

        /**
         * Create a d-sprites environment.
         * @param path the path in which the d-sprites dataset is stored
         * @param granularity the size of each superpixel used by the agent to perceive the world.
         * @param repeat the number of times an action is repeated
         * @return the environment
         */
        static std::unique_ptr<DisentangleSpritesEnv> create(const std::string &path, int granularity, int repeat);

        //
        // Constructor
        //
        /**
         * Construct a lake environment.
         * @param file the name of the file from which the environment should be loaded
         * @param granularity the size of each superpixel used by the agent to perceive the world.
         * @param repeat the number of times an action is repeated
         */
        explicit DisentangleSpritesEnv(const std::string &file, int granularity, int repeat);

    public:
        //
        //
        //
        /**
         * The actions available to the agent in the dSprites environment.
         */
        enum Action: int {
            UP    = 0,
            DOWN  = 1,
            LEFT  = 2,
            RIGHT = 3,
        };

    public:
        //
        // Implementation of the methods of the Environment class
        //

        /**
         * Load a Pytorch tensor from a file.
         * @param file_name the file's name from which the tensor must be loaded
         * @return the loaded tensor
         */
        static torch::Tensor loadTensorFromFile(const std::string &file_name);

        /**
         * Reset the hidden state of the environment to a random value.
         */
        void reset_hidden_state();

        /**
         * Return the observation corresponding to the current hidden state of the environment.
         * @return the current observation.
         */
        [[nodiscard]] torch::Tensor current_observation() const;

        /**
         * Return the observation corresponding to the current hidden state of the environment.
         * @return the current observation.
         */
        [[nodiscard]] torch::Tensor current_image() const;

        /**
         * Reset the environment to its initial state.
         */
        torch::Tensor reset() override;

        /**
         * Execute an action in the environment.
         * @param action the action to be executed
         * @return the observation made after executing the action
         */
        torch::Tensor execute(int action) override;

        /**
         * Display the environment.
         */
        void print() override;

        /**
         * Getter.
         * @return the number of actions available to the agent
         */
        [[nodiscard]] int actions() const override;

        /**
         * Getter.
         * @return the number of states in the environment
         */
        [[nodiscard]] int states() const override;

        /**
         * Getter.
         * @return the number of observations in the environment
         */
        [[nodiscard]] int observations() const override;

        /**
         * Getter.
         * @return the true likelihood mapping
         */
        [[nodiscard]] torch::Tensor A() const override;

        /**
         * Getter.
         * @return the true transition mapping
         */
        [[nodiscard]] torch::Tensor B() const override;

        /**
         * Getter.
         * @return the true initial hidden states
         */
        [[nodiscard]] torch::Tensor D() const override;

        /**
         * Getter.
         * @param advanced should the prior preferences be advanced?
         * @return the prior preferences over observations
         */
        [[nodiscard]] torch::Tensor pref_states(bool advanced) const override;

        /**
         * Getter.
         * @return the prior preferences over observations
         */
        [[nodiscard]] torch::Tensor pref_obs() const override;

        /**
         * Getter.
         * @return true if the agent solved the environment false otherwise.
         */
        [[nodiscard]] bool solved() const override;

        /**
         * Getter.
         * @return environment's type.
         */
        [[nodiscard]] EnvType type() const override;

    public:
        /**
         * Getter.
         * @param state the state whose index must be calculated
         * @return the index of the states sent as parameter.
         */
        [[nodiscard]] int get_obs_id(const torch::Tensor &state) const;

        /**
         * Getter.
         * @return the index of the current image.
         */
        [[nodiscard]] int current_img_id() const;

        /**
         * Getter.
         * @return the reward obtained by the agent during the last run.
         */
        [[nodiscard]] double reward_obtained() const;

    private:
        /**
         * Getter.
         * @param i the index of the state to recover.
         * @return the vector representation corresponding to the index "i".
         */
        [[nodiscard]] torch::Tensor getStateFromId(int i) const;

        /**
         * Compute the reward obtained by the agent in the current state,
         * if the blob in the image is a square.
         * @return the reward.
         */
        [[nodiscard]] double compute_square_reward() const;

        /**
         * Compute the reward obtained by the agent in the current state,
         * if the blob in the image is an ellipse or an heart.
         * @return the reward.
         */
        [[nodiscard]] double compute_non_square_reward() const;

        /**
         * Simulate the execution of an action in the environment but does not modify the environment state.
         * @param action the action to perform
         * @param pos the current state of the environment
         * @return the new state of the environment
         */
        [[nodiscard]] torch::Tensor execute(int action, const torch::Tensor &pos) const;

    private:
        torch::Tensor images;  // D-sprites' images
        torch::Tensor classes; // D-sprites' classes

        torch::Tensor state; // Current state of the system
        long s_dim; // Number of dimensions in the latent space
        std::vector<int> s_sizes; // Number of values for each dimension of the latent space
        torch::Tensor s_bases_obs; // Basis vectors used to get the index of the current
                                   // observation from the current state.
        torch::Tensor s_bases_img; // Basis vectors used to get the index of the image
                                   // corresponding to the current state.

        bool need_reset;
        int granularity;
        int n_pos;
        double last_r;
        double repeats;
        int frame_id;
        int max_episode_length;
    };

}

#endif //HOMING_PIGEON_DISENTANGLE_SPRITES_ENV_H
