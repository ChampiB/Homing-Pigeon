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

    auto boolean1 = where(t3 == 0, true, false); // Element wise: (elem == 0) ? true : false
    std::cout << boolean1 << std::endl; // Display: [1,0,0,0,0,0,0,0,0,0]

    auto boolean2 = where(t3 < 5, true, false); // Element wise: (elem < 0) ? true : false
    std::cout << boolean2 << std::endl; // Display: [1,1,1,1,1,0,0,0,0,0]

    Tensor t4 = t3.index({boolean2}); // Select the elements corresponding to the mask boolean2
    std::cout << t4 << std::endl; // Display: [0,1,2,3,4]

    Tensor t5 = torch::arange(1,10); // Create a 1D vector: [1,...,9]
    Tensor v5 = t5.view({3, 3}); // Reshape the vector into a 3x3 matrix
    std::cout << v5.index({0,None}) << std::endl; // Display: [1,2,3]
    std::cout << v5.index({1,None}) << std::endl; // Display: [4,5,6]
    std::cout << v5.index({2,None}) << std::endl; // Display: [7,8,9]

    Tensor t6 = tensor({0.1, 0.3, 0.7, 0.4, 0.2, 0.3}); // Create a tensor with specific values
    std::cout << t6 << std::endl; // Display: [0.1,0.3,0.7,0.4,0.2,0.3]

    Tensor t7 = tensor({0.0,1.0,3.14,42.0}).view({2,2}); // Create a 2x2 matrix with specific values
    std::cout << t7[0][0] << std::endl;       // Display: 0
    std::cout << t7[0][0].dim() << std::endl; // Display: 0

    auto access = t7.accessor<double,2>(); // assert t7 is 2-dimensional and holds floats.
    double trace = 0;
    for(int i = 0; i < access.size(0); i++) {
        for(int j = 0; j < access.size(1); j++) {
            trace += access[i][j]; // use the accessor t7 to get tensor data.
            access[i][j] = 3; // use the accessor foo_a to get tensor data.
        }
    }
    std::cout << trace << std::endl; // Display: 46.14
    std::cout << t7 << std::endl; // Display: [[3,3],[3,3]]

    Tensor t8 = torch::arange(0, 24); // Create a 1D vector: [0,...,23]
    Tensor v8 = t8.view({2, 3, 4}); // Reshape the vector into a 2x3x4 tensor
    std::cout << v8 << std::endl; // Display:  0   1   2   3
                                  //           4   5   6   7
                                  //           8   9  10  11
                                  //
                                  //           12  13  14  15
                                  //           16  17  18  19
                                  //           20  21  22  23
    std::cout << v8.size(0) << std::endl; // Display:  2
    std::cout << v8.size(1) << std::endl; // Display:  3
    std::cout << v8.size(2) << std::endl; // Display:  4

    // Indexing works like {matrix_index, row_index, column_index}
    // Nones are ignored {matrix_index, row_index} = {matrix_index, None, None, None, None, None, row_index}
    // Ellipsis select a slice, i.e., all indices of along a dimension,
    //                          e.g., {matrix_index, row_index, Ellipsis} select the entire row in a specific matrix
    std::cout << v8.index({0,0,1}) << std::endl; // Display:  1
    std::cout << v8.index({0,1}) << std::endl; // Display:  4 5 6 7
    std::cout << v8.index({0,Ellipsis,1}) << std::endl; // Display:  1 5 9

    std::cout << torch::narrow(v8, 0, 0, 1) << std::endl;
    // Display:  0   1   2   3
    //           4   5   6   7
    //           8   9  10  11
    std::cout << torch::narrow(v8, 2, 1, 1) << std::endl;
    // Display:  1  13
    //           5  17
    //           9  21
}
