//
// Created by Theophile Champion on 28/11/2020.
//

#include "DisentangleSpritesEnv.h"
#include <random>
#include <api/API.h>
#include <math/Ops.h>
#include <iostream>
#include <torch/script.h>

using namespace torch;
using namespace torch::jit;
using namespace hopi::api;
using namespace hopi::math;

namespace hopi::environments {

    std::unique_ptr<DisentangleSpritesEnv> DisentangleSpritesEnv::create(const std::string &file, int granularity, int repeat) {
        return std::make_unique<DisentangleSpritesEnv>(file, granularity, repeat);
    }

    DisentangleSpritesEnv::DisentangleSpritesEnv(const std::string &path, int gran, int rep) {
        // Load the d-sprites images
        std::vector<torch::Tensor> tensors;
        for (int i = 0; i < 10; ++i) {
            std::string file_name = path + "images" + std::to_string(i) + ".pt";
            torch::Tensor tensor = loadTensorFromFile(file_name);
            tensors.push_back(tensor);
        }
        images = torch::cat(tensors);

        // Initialize class attributes
        granularity = gran;
        n_pos = 32 / gran;
        last_r = 0.0;
        frame_id = 0;
        max_episode_length = 50;
        repeats = rep;
        need_reset = false;

        s_sizes = std::vector<int>({1, 3, 6, 40, 32, 32}); // Number of values for each possible latent dimension.
        s_dim = (long) s_sizes.size();
        s_bases_obs = API::tensor({0, n_pos * (n_pos + 1), 0, 0, n_pos + 1, 1});
        s_bases_img = API::tensor({737280, 245760, 40960, 1024, 32, 1});
        state = API::zeros({s_dim});
    }

    torch::Tensor DisentangleSpritesEnv::loadTensorFromFile(const string &file_name) {
        // Open file
        std::ifstream classes_file(file_name);

        // Get file's length
        classes_file.seekg(0, std::ios::end);
        long length = classes_file.tellg();
        classes_file.seekg(0, std::ios::beg);

        // Load tensor from file
        std::vector<char> buffer(length);
        classes_file.read(&buffer[0], length);
        torch::Tensor tensor = torch::pickle_load(buffer).toTensor();
        return tensor;
    }

    void DisentangleSpritesEnv::reset_hidden_state() {
        std::default_random_engine generator(std::random_device{}());

        state = API::zeros({s_dim});
        for (int i = 0; i < s_dim; ++i) {
            std::uniform_int_distribution<int> distribution(0, s_sizes[i] - 1);
            state[i] = distribution(generator);
        }
    }

    torch::Tensor DisentangleSpritesEnv::reset() {
        state = API::zeros({s_dim});
        last_r = 0.0;
        frame_id = 0;
        need_reset = false;
        reset_hidden_state();
        return current_observation();
    }

    int DisentangleSpritesEnv::get_obs_id(const torch::Tensor &s) const {
        Tensor s2 = s.clone();

        s2[4] = s2[4].item<int>() / granularity; // Divide x and y position by granularity.
        s2[5] = s2[5].item<int>() / granularity; // Divide x and y position by granularity.
        return torch::dot(s2, s_bases_obs).item<int>();
    }

    int DisentangleSpritesEnv::current_img_id() const {
        return torch::dot(state, s_bases_img).item<int>();
    }

    torch::Tensor DisentangleSpritesEnv::current_observation() const {
        return Ops::one_hot(observations(), get_obs_id(state));
    }

    torch::Tensor DisentangleSpritesEnv::current_image() const {
        return images[current_img_id()];
    }

    int DisentangleSpritesEnv::actions() const {
        return 4;
    }

    int DisentangleSpritesEnv::observations() const {
        return (n_pos + 1) * n_pos * 3;
    }

    int DisentangleSpritesEnv::states() const {
        return observations();
    }

    torch::Tensor DisentangleSpritesEnv::pref_states(bool advanced) const {
        if (advanced)
            throw std::invalid_argument("Advanced prior are not supported in graph environment.");
        return Ops::uniform({states()});
    }

    Tensor DisentangleSpritesEnv::A() const {
        double noise = 0.01;
        double epsilon = noise / (observations() - 1);
        Tensor res = API::full({observations(), observations()}, epsilon);

        for (int i = 0; i < observations(); ++i) {
            res[i][i] = 1 - noise;
        }
        return res;
    }

    Tensor DisentangleSpritesEnv::B() const {
        int n_states = states();
        int n_actions = actions();
        Tensor B = API::full({n_states, n_states, n_actions}, 0.1 / (n_states - 1));

        for (int i = 0; i < n_states; ++i) {
            Tensor cur_state = getStateFromId(i);
            for (int k = 0; k < n_actions; ++k) {
                auto dest_state = execute(k, cur_state);
                B[get_obs_id(dest_state)][i][k] = 0.9;
            }
        }
        return B;
    }

    Tensor DisentangleSpritesEnv::D() const {
        Tensor D = API::full({states()}, 0.1 / (states() - 1));

        D[get_obs_id(state)] = 0.9;
        return D;
    }

    EnvType DisentangleSpritesEnv::type() const {
        return D_SPRITES;
    }

    torch::Tensor DisentangleSpritesEnv::execute(int action, const torch::Tensor &pos) const {
        int x_pos = 4;
        int y_pos = 5;
        torch::Tensor res = pos.clone();

        for (int i = 0; i < repeats; ++i) {
            if (res[y_pos].item<int>() >= 32) // If the agent cross the bottom of the image
                return res;                   // Then the agent is in an absorbing state
            switch (action) {
                case UP:
                    if (res[y_pos].item<int>() - 1 >= 0)
                        res[y_pos] -= 1;
                    break;
                case DOWN:
                    if (res[y_pos].item<int>() + 1 <= 32)
                        res[y_pos] += 1;
                    break;
                case LEFT:
                    if (res[x_pos].item<int>() - 1 >= 0)
                        res[x_pos] -= 1;
                    break;
                case RIGHT:
                    if (res[x_pos].item<int>() + 1 < 32)
                        res[x_pos] += 1;
                    break;
                default:
                    assert(false && "D-SpritesEnv::execute, unsupported action.");
            }
        }
        return res;
    }

    torch::Tensor DisentangleSpritesEnv::execute(int action) {
        // Increase the frame index, that count the number of frames since
        // the beginning of the episode.
        frame_id += 1;

        // Simulate the action requested by the user.
        state = execute(action, state);

        // If the object crossed the bottom line, then:
        // compute the reward, generate a new image .
        if (state[5].item<double>() >= 32) {
            if (state[1].item<double>() < 0.5) {
                last_r = compute_square_reward();
            } else {
                last_r = compute_non_square_reward();
            }
            need_reset = true;
        }

        // Make sure the environment is reset if the maximum number of steps in
        // the episode has been reached.
        if (frame_id >= max_episode_length) {
            need_reset = true;
            last_r = -1;
        }
        return current_observation();
    }

    void DisentangleSpritesEnv::print() {
        std::map<int, string> colors = {{0, "white"}};
        std::map<int, string> shapes = {
                {0, "square"},
                {1, "ellipse"},
                {2, "heart"}
        };

        std::cout << "Color: " << colors[state[0].item<int>()] << std::endl;
        std::cout << "Shape: " << shapes[state[1].item<int>()] << std::endl;
        std::cout << "Scale: " << state[2].item<int>() << std::endl;
        std::cout << "Orientation: " << state[3].item<int>() << std::endl;
        std::cout << "True X position: " << state[4].item<int>() << std::endl;
        std::cout << "True Y position: " << state[5].item<int>() << std::endl;
        std::cout << "Perceived X position: " << state[4].item<int>() / granularity << std::endl;
        std::cout << "Perceived Y position: " << state[5].item<int>() / granularity << std::endl;
        std::cout << "Last reward: " << last_r << std::endl;
        std::cout << std::endl;

        // Add the current image to the video, if state is valid, i.e. if y_pos < 32
        if (state[5].item<double>() < 32) {
            Tensor image = current_image().clone();
            image *= 255;
            cv::Mat cv_image(cv::Size{64, 64}, CV_8U, image.data_ptr<uchar>());
            int dims[] = {3, 64, 64};
            cv::Mat cv_image_out(3, dims, CV_32F, cv::Scalar(0));
            cv::cvtColor(cv_image, cv_image_out, cv::COLOR_GRAY2RGB, 3);
            cv::imshow("d-sprites", cv_image_out);
            cv::waitKey(1);
        }
    }

    torch::Tensor DisentangleSpritesEnv::pref_obs() const {
        Tensor C = API::zeros({observations()});
        Tensor bottom_state = API::tensor({0, 0, 0, 0, 0, 32});

        for (int shape = 0; shape < 3; ++shape) {
            bottom_state[1] = shape;
            int start_pos = (shape == 0) ? 1 : 0; // If (shape == square) then 1 else 0
            int end_pos   = (shape == 0) ? n_pos : n_pos - 1; // If (shape == square) then n_pos else n_pos - 1
            for (int x_pos = start_pos; x_pos < end_pos; ++x_pos) {
                bottom_state[4] = x_pos * granularity;
                C[get_obs_id(bottom_state)] = -5;
            }
            if (shape == 0) {
                bottom_state[4] = 0;
                C[get_obs_id(bottom_state)] = 5;
            } else {
                bottom_state[4] = (n_pos - 1) * granularity;
                C[get_obs_id(bottom_state)] = 5;
            }
        }
        return C;
    }

    bool DisentangleSpritesEnv::solved() const {
        return need_reset;
    }

    double DisentangleSpritesEnv::reward_obtained() const {
        return last_r;
    }

    double DisentangleSpritesEnv::compute_square_reward() const {
        auto x_pos = state[4].item<double>();

        if (x_pos > 15)
            return (15.0 - x_pos) / 16.0;
        else
            return (16.0 - x_pos) / 16.0;
    }

    double DisentangleSpritesEnv::compute_non_square_reward() const {
        auto x_pos = state[4].item<double>();

        if (x_pos > 15)
            return (x_pos - 15.0) / 16.0;
        else
            return (x_pos - 16.0) / 16.0;
    }

    torch::Tensor DisentangleSpritesEnv::getStateFromId(int i) const {
        torch::Tensor cur_state = API::tensor({0, 0, 0, 0, 0, 0});

        cur_state[1] = i / (n_pos * (n_pos + 1));
        i %= (n_pos * (n_pos + 1));
        cur_state[4] = granularity * (i / (n_pos + 1));
        cur_state[5] = granularity * (i % (n_pos + 1));
        return cur_state;
    }

}
