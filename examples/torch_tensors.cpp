#include <torch/torch.h>
#include <iostream>

using namespace torch;
using namespace torch::indexing;

int main() {
    Tensor t1 = torch::eye(3); // Create identity matrix
    Tensor t2 = torch::eye(3); // Create identity matrix

    std::cout << t1 << std::endl; // Display a tensor
    std::cout << t1 + t2 << std::endl; // Element wise addition
    std::cout << t1 * t2 << std::endl; // Element wise multiplication
    std::cout << t1 - t2 << std::endl; // Element wise subtraction
    std::cout << t1.sum() << std::endl; // Sum all elements

    Tensor t3 = torch::arange(0, 10); // Create a 1D vector: [0,...,9]
    Tensor v3 = t3.view({2, 5}); // Reshape the vector into a 2x5 matrix

    std::cout << t3 << std::endl; // Display: [0,...,9]
    std::cout << v3 << std::endl; // Display: [[0,1,2,3,4], [5,6,7,8,9]]
    std::cout << v3.sum(std::vector<int64_t>{0}) << std::endl; // Sum over the 0-th dimension: [5,7,9,11,13]
    std::cout << v3.sum(std::vector<int64_t>{1}) << std::endl; // Sum over the 1-th dimension: [10,35]

    auto boolean1 = torch::where(t3 == 0, true, false); // Element wise: (elem == 0) ? true : false
    std::cout << boolean1 << std::endl; // Display: [1,0,0,0,0,0,0,0,0,0]

    auto boolean2 = torch::where(t3 < 5, true, false); // Element wise: (elem < 0) ? true : false
    std::cout << boolean2 << std::endl; // Display: [1,1,1,1,1,0,0,0,0,0]

    Tensor t4 = t3.index({boolean2}); // Select the elements corresponding to the mask boolean2
    std::cout << t4 << std::endl; // Display: [0,1,2,3,4]

    Tensor t5 = torch::arange(1,10); // Create a 1D vector: [1,...,9]
    Tensor v5 = t5.view({3, 3}); // Reshape the vector into a 3x3 matrix
    std::cout << v5.index({0,None}) << std::endl; // Display: [1,2,3]
    std::cout << v5.index({1,None}) << std::endl; // Display: [4,5,6]
    std::cout << v5.index({2,None}) << std::endl; // Display: [7,8,9]

    Tensor t6 = torch::tensor({0.1, 0.3, 0.7, 0.4, 0.2, 0.3}); // Create a tensor with specific values
    std::cout << t6 << std::endl; // Display: [0.1,0.3,0.7,0.4,0.2,0.3]
}