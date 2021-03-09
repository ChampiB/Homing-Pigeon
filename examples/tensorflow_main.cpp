#include <iostream>
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/client/client_session.h"
#include <chrono>
#include "dnn/LayersStackBuilder.h"

using namespace std;
using namespace chrono;
using namespace tensorflow;
using namespace tensorflow::ops;

int main()
{
    int image_side = 150;
    int image_channels = 3;
    LayersStackBuilder model(image_side, image_channels);
    Status s = model.CreateGraphForImage(true);
    TF_CHECK_OK(s);

    std::string train_folder = "../examples/cats_and_dogs/train";
    int batch_size = 20;

    std::vector<Tensor> image_batches, label_batches, valid_images, valid_labels;
    //Label: cat=0, dog=1
    s = model.ReadBatches(train_folder, {make_pair("cats", 0), make_pair("dogs", 1)}, batch_size, image_batches, label_batches);
    TF_CHECK_OK(s);

    std::string validation_folder = "../examples/cats_and_dogs/validation";
    s = model.ReadBatches(validation_folder, {make_pair("cats", 0), make_pair("dogs", 1)}, batch_size, valid_images, valid_labels);
    TF_CHECK_OK(s);

    //CNN model
    int filter_side = 3;
    s = model.CreateGraphForCNN(filter_side);
    TF_CHECK_OK(s);
    s = model.CreateOptimizationGraph(0.0001f); //input is learning rate
    TF_CHECK_OK(s);

    //Run initialization
    s = model.Initialize();
    TF_CHECK_OK(s);

    size_t num_batches = image_batches.size();
    assert(num_batches == label_batches.size());
    size_t valid_batches = valid_images.size();
    assert(valid_batches == valid_labels.size());

    int num_epochs = 20;
    //Epoch / Step loops
    for(int epoch = 0; epoch < num_epochs; epoch++)
    {
        std::cout << "Epoch " << epoch+1 << "/" << num_epochs << ":";
        auto t1 = high_resolution_clock::now();
        float loss_sum = 0;
        float accuracy_sum = 0;
        for(int b = 0; b < num_batches; b++)
        {
            std::vector<float> results;
            float loss;
            s = model.TrainCNN(image_batches[b], label_batches[b], results, loss);
            TF_CHECK_OK(s);
            loss_sum += loss;
            accuracy_sum += accumulate(results.begin(), results.end(), 0.f) / results.size();
            std::cout << ".";
        }
        std::cout << std::endl << "Validation:";
        float validation_sum = 0;
        for(int c = 0; c < valid_batches; c++)
        {
            std::vector<float> results;
            s = model.ValidateCNN(valid_images[c], valid_labels[c], results);
            TF_CHECK_OK(s);
            validation_sum += accumulate(results.begin(), results.end(), 0.f) / results.size();
            std::cout << ".";

        }
        auto t2 = high_resolution_clock::now();
        std::cout << std::endl << "Time: " << duration_cast<seconds>(t2-t1).count() << " seconds ";
        std::cout << "Loss: " << loss_sum/num_batches << " Results accuracy: " << accuracy_sum/num_batches << " Validation accuracy: " << validation_sum/valid_batches << std::endl;
    }
    //testing the model
    s = model.CreateGraphForImage(false);//rebuild the model without unstacking
    TF_CHECK_OK(s);
    std::string test_folder = "../examples/cats_and_dogs/test";
    std::vector<std::pair<Tensor, float>> all_files_tensors;
    s = model.ReadFileTensors(test_folder, {make_pair("cats", 0), make_pair("dogs", 1)}, all_files_tensors);
    TF_CHECK_OK(s);
    //test a few images
    int count_images = 20;
    int count_success = 0;
    for(int i = 0; i < count_images; i++)
    {
        std::pair<Tensor, float> p = all_files_tensors[i];
        int result;
        s = model.Predict(p.first, result);
        TF_CHECK_OK(s);
        std::cout << "Test number: " << i + 1 << " predicted: " << result << " actual is: " << p.second << std::endl;
        if(result == (int)p.second)
            count_success++;
    }
    std::cout << "total successes: " << count_success << " out of " << count_images << std::endl;
    return 0;
}
