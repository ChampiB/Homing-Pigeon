//
// Created by tmac3 on 08/03/2021.
//

#ifndef HOMING_PIGEON_LAYERS_STACK_BUILDER_H
#define HOMING_PIGEON_LAYERS_STACK_BUILDER_H

#include <iostream>
#include <map>
#include <fstream>
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/cc/tools/freeze_saved_model.h"

class LayersStackBuilder {
public:
    LayersStackBuilder(int side, int channels);
    tensorflow::Status CreateGraphForImage(bool unstack);
    tensorflow::Status ReadTensorFromImageFile(
            std::string& file_name,
            tensorflow::Tensor& outTensor
    );
    tensorflow::Status ReadFileTensors(
            std::string& folder_name,
            const std::vector<std::pair<std::string, float>>& v_folder_label,
            std::vector<std::pair<tensorflow::Tensor, float>>& file_tensors
    );
    tensorflow::Status ReadBatches(
            std::string& folder_name,
            const std::vector<std::pair<std::string, float>>& v_folder_label,
            int batch_size,
            std::vector<tensorflow::Tensor>& image_batches,
            std::vector<tensorflow::Tensor>& label_batches
    );
    static tensorflow::Input XavierInit(
            const tensorflow::Scope& scope,
            int in_chan,
            int out_chan,
            int filter_side = 0
    );
    tensorflow::Input AddConvLayer(
            const std::string& idx,
            const tensorflow::Scope& scope,
            int in_channels,
            int out_channels,
            int filter_side,
            tensorflow::Input input
    );
    tensorflow::Input AddDenseLayer(
            const std::string& idx,
            const tensorflow::Scope& scope,
            int in_units,
            int out_units,
            bool bActivation,
            tensorflow::Input input
    );
    tensorflow::Status CreateGraphForCNN(int filter_side);
    tensorflow::Status CreateOptimizationGraph(float learning_rate);
    tensorflow::Status Initialize();
    tensorflow::Status TrainCNN(
            tensorflow::Tensor& image_batch,
            tensorflow::Tensor& label_batch,
            std::vector<float>& results,
            float& loss
    );
    tensorflow::Status ValidateCNN(
            tensorflow::Tensor& image_batch,
            tensorflow::Tensor& label_batch,
            std::vector<float>& results
    );
    tensorflow::Status Predict(tensorflow::Tensor& image, int& result);
    tensorflow::Status FreezeSave(std::string& file_name);
    tensorflow::Status LoadSavedModel(std::string& file_name);
    tensorflow::Status PredictFromFrozen(tensorflow::Tensor& image, int& result);
    tensorflow::Status WriteBatchToImageFiles(
            tensorflow::Tensor& image_batch,
            const std::string& folder_name,
            const std::string& image_name
    );
private:
    tensorflow::Scope i_root; //graph for loading images into tensors
    const int image_side; //assuming square picture
    const int image_channels; //RGB
    //load image vars
    tensorflow::Output file_name_var;
    tensorflow::Output image_tensor_var;
    //training and validating the CNN
    tensorflow::Scope t_root; //graph
    std::unique_ptr<tensorflow::ClientSession> t_session;
    std::unique_ptr<tensorflow::Session> f_session;
    //CNN vars
    tensorflow::Output input_batch_var;
    std::string input_name = "input";
    tensorflow::Output input_labels_var;
    tensorflow::Output drop_rate_var; //use real drop rate in training and 1 in validating
    std::string drop_rate_name = "drop_rate";
    tensorflow::Output skip_drop_var; //use 0 in trainig and 1 in validating
    std::string skip_drop_name = "skip_drop";
    tensorflow::Output out_classification;
    std::string out_name = "output_classes";
    tensorflow::Output logits;
    //Network maps
    std::map<std::string, tensorflow::Output> m_vars;
    std::map<std::string, tensorflow::TensorShape> m_shapes;
    std::map<std::string, tensorflow::Output> m_assigns;
    //Loss variables
    std::vector<tensorflow::Output> v_weights_biases;
    std::vector<tensorflow::Operation> v_out_grads;
    tensorflow::Output out_loss_var;
};

#endif //HOMING_PIGEON_LAYERS_STACK_BUILDER_H
