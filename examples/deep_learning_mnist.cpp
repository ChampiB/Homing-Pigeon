#include <torch/torch.h>
#include <iostream>

using namespace torch;

int main() {
    // Load the mnist dataset
    int kBatchSize = 30;
    auto dataset = data::datasets::MNIST("./mnist")
            .map(data::transforms::Normalize<>(0.5, 0.5))
            .map(data::transforms::Stack<>());
    const int64_t batches_per_epoch = std::ceil(dataset.size().value() / static_cast<double>(kBatchSize));
    auto data_loader = data::make_data_loader(
            std::move(dataset),
            data::DataLoaderOptions().batch_size(kBatchSize).workers(2));

    // Create the classifier
    nn::Sequential classifier(
            // Layer 1
            nn::Conv2d(nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false)),
            nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
            // Layer 2
            nn::Conv2d(nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).bias(false)),
            nn::BatchNorm2d(128),
            nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
            // Layer 3
            nn::Conv2d(nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).bias(false)),
            nn::BatchNorm2d(256),
            nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
            // Layer 4
            nn::Conv2d(nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).bias(false)),
            nn::Sigmoid());

    // Create the optimizer
    optim::Adam optimizer(classifier->parameters(), optim::AdamOptions(5e-4));

    // Learning loop
    long kNumberOfEpochs = 30;
    for (long epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
        long batch_index = 0;
        for (data::Example<>& batch : *data_loader) {
            // Train the classifier
            classifier->zero_grad();
            Tensor images = batch.data;
            Tensor labels = torch::empty(batch.data.size(0)).uniform_(0.8, 1.0);
            Tensor output = classifier->forward(images);
            Tensor loss = binary_cross_entropy(output, labels);
            loss.backward();
            optimizer.step();
            std::printf("\r[%2ld/%2ld][%3ld/%3ld] loss: %.4f",
                        epoch, kNumberOfEpochs, ++batch_index, batches_per_epoch, loss.item<float>());
        }
    }
}