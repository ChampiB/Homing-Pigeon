//
// Created by tmac3 on 08/03/2021.
//

#include "LayersStackBuilder.h"

#include <memory>
#include <utility>

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

LayersStackBuilder::LayersStackBuilder(int side, int channels)
    : i_root(tensorflow::Scope::NewRootScope()),
      t_root(tensorflow::Scope::NewRootScope()),
      image_side(side),
      image_channels(channels) {}

Status LayersStackBuilder::CreateGraphForImage(bool unstack)
{
    file_name_var = Placeholder(i_root.WithOpName("input"), DT_STRING);
    auto file_reader = ops::ReadFile(i_root.WithOpName("file_reader"), file_name_var);

    auto image_reader = DecodeJpeg(i_root.WithOpName("jpeg_reader"), file_reader, DecodeJpeg::Channels(image_channels));

    auto float_caster = Cast(i_root.WithOpName("float_caster"), image_reader, DT_FLOAT);
    auto dims_expander = ExpandDims(i_root.WithOpName("dim"), float_caster, 0);
    auto resized = ResizeBilinear(i_root.WithOpName("size"), dims_expander, Const(i_root, {image_side, image_side}));
    auto div = Div(i_root.WithOpName("normalized"), resized, {255.f});
    if(unstack)
    {
        auto output_list = Unstack(i_root.WithOpName("fold"), div, 1);
        image_tensor_var = output_list.output[0];
    }
    else
        image_tensor_var = div;
    return i_root.status();
}

Status LayersStackBuilder::ReadTensorFromImageFile(string& file_name, Tensor& outTensor)
{
    if(!i_root.ok())
        return i_root.status();
    if (!absl::EndsWith(file_name, ".jpg") && !absl::EndsWith(file_name, ".jpeg"))
    {
        return errors::InvalidArgument("Image must be jpeg encoded");
    }
    vector<Tensor> out_tensors;
    ClientSession session(i_root);
    TF_CHECK_OK(session.Run({{file_name_var, file_name}}, {image_tensor_var}, &out_tensors));
    outTensor = out_tensors[0]; // shallow copy
    return Status::OK();
}

Status LayersStackBuilder::ReadFileTensors(string& base_folder_name, const vector<pair<string, float>>& v_folder_label, vector<pair<Tensor, float>>& file_tensors)
{
    //validate the folder
    Env* penv = Env::Default();
    TF_RETURN_IF_ERROR(penv->IsDirectory(base_folder_name));
    //get the files
    bool b_shuffle = false;
    for(const auto& p: v_folder_label)
    {
        string folder_name = io::JoinPath(base_folder_name, p.first);
        TF_RETURN_IF_ERROR(penv->IsDirectory(folder_name));
        vector<string> file_names;
        TF_RETURN_IF_ERROR(penv->GetChildren(folder_name, &file_names));
        for(const string& file: file_names)
        {
            string full_path = io::JoinPath(folder_name, file);
            Tensor i_tensor;
            TF_RETURN_IF_ERROR(ReadTensorFromImageFile(full_path, i_tensor));
            size_t s = file_tensors.size();
            if(b_shuffle)
            {
                //suffle the images
                int i = rand() % s;
                file_tensors.emplace(file_tensors.begin()+i, make_pair(i_tensor, p.second));
            }
            else
                file_tensors.emplace_back(i_tensor, p.second);
        }
        b_shuffle = true;
    }
    return Status::OK();
}

Status LayersStackBuilder::ReadBatches(string& base_folder_name, const vector<pair<string, float>>& v_folder_label, int batch_size, vector<Tensor>& image_batches, vector<Tensor>& label_batches)
{
    vector<pair<Tensor, float>> all_files_tensors;
    TF_RETURN_IF_ERROR(ReadFileTensors(base_folder_name, v_folder_label, all_files_tensors));
    auto start_i = all_files_tensors.begin();
    auto end_i = all_files_tensors.begin()+batch_size;
    size_t batches = all_files_tensors.size()/batch_size;
    if(batches*batch_size < all_files_tensors.size())
        batches++;
    for(int b = 0; b < batches; b++)
    {
        if(end_i > all_files_tensors.end())
            end_i = all_files_tensors.end();
        vector<pair<Tensor, float>> one_batch(start_i, end_i);
        //need to break the pairs
        vector<Input> one_batch_image, one_batch_lbl;
        for(const auto& p: one_batch)
        {
            one_batch_image.emplace_back(p.first);
            Tensor t(DT_FLOAT, TensorShape({1}));
            t.scalar<float>()(0) = p.second;
            one_batch_lbl.emplace_back(t);
        }
        InputList one_batch_inputs(one_batch_image);
        InputList one_batch_labels(one_batch_lbl);
        Scope root = Scope::NewRootScope();
        auto stacked_images = Stack(root, one_batch_inputs);
        auto stacked_labels = Stack(root, one_batch_labels);
        TF_CHECK_OK(root.status());
        ClientSession session(root);
        vector<Tensor> out_tensors;
        TF_CHECK_OK(session.Run({}, {stacked_images, stacked_labels}, &out_tensors));
        image_batches.push_back(out_tensors[0]);
        label_batches.push_back(out_tensors[1]);
        start_i = end_i;
        if(start_i == all_files_tensors.end())
            break;
        end_i = start_i+batch_size;
    }
    return Status::OK();
}

Input LayersStackBuilder::XavierInit(const Scope& scope, int in_chan, int out_chan, int filter_side)
{
    float std;
    Tensor t;
    if (filter_side == 0)
    { //Dense
        std = sqrt(6.f/((float)(in_chan+out_chan)));
        Tensor ts(DT_INT64, {2});
        auto v = ts.vec<int64>();
        v(0) = in_chan;
        v(1) = out_chan;
        t = ts;
    }
    else
    { //Conv
        std = sqrt(6.f / ((float)(filter_side * filter_side * (in_chan + out_chan))));
        Tensor ts(DT_INT64, {4});
        auto v = ts.vec<int64>();
        v(0) = filter_side;
        v(1) = filter_side;
        v(2) = in_chan;
        v(3) = out_chan;
        t = ts;
    }
    auto rand = RandomUniform(scope, t, DT_FLOAT);
    return Multiply(scope, Sub(scope, rand, 0.5f), std*2.f);
}

Input LayersStackBuilder::AddConvLayer(const string& idx, const Scope& scope, int in_channels, int out_channels, int filter_side, Input input)
{
    TensorShape sp({filter_side, filter_side, in_channels, out_channels});
    m_vars["W"+idx] = Variable(scope.WithOpName("W"), sp, DT_FLOAT);
    m_shapes["W"+idx] = sp;
    m_assigns["W"+idx+"_assign"] = Assign(scope.WithOpName("W_assign"), m_vars["W"+idx], XavierInit(scope, in_channels, out_channels, filter_side));
    sp = {out_channels};
    m_vars["B"+idx] = Variable(scope.WithOpName("B"), sp, DT_FLOAT);
    m_shapes["B"+idx] = sp;
    m_assigns["B"+idx+"_assign"] = Assign(scope.WithOpName("B_assign"), m_vars["B"+idx], Input::Initializer(0.f, sp));
    auto conv = Conv2D(scope.WithOpName("Conv"), std::move(input), m_vars["W"+idx], {1, 1, 1, 1}, "SAME");
    auto bias = BiasAdd(scope.WithOpName("Bias"), conv, m_vars["B"+idx]);
    auto relu = Relu(scope.WithOpName("Relu"), bias);
    return MaxPool(scope.WithOpName("Pool"), relu, {1, 2, 2, 1}, {1, 2, 2, 1}, "SAME");
}

Input LayersStackBuilder::AddDenseLayer(const string& idx, const Scope& scope, int in_units, int out_units, bool bActivation, Input input)
{
    TensorShape sp = {in_units, out_units};
    m_vars["W"+idx] = Variable(scope.WithOpName("W"), sp, DT_FLOAT);
    m_shapes["W"+idx] = sp;
    m_assigns["W"+idx+"_assign"] = Assign(scope.WithOpName("W_assign"), m_vars["W"+idx], XavierInit(scope, in_units, out_units));
    sp = {out_units};
    m_vars["B"+idx] = Variable(scope.WithOpName("B"), sp, DT_FLOAT);
    m_shapes["B"+idx] = sp;
    m_assigns["B"+idx+"_assign"] = Assign(scope.WithOpName("B_assign"), m_vars["B"+idx], Input::Initializer(0.f, sp));
    auto dense = Add(scope.WithOpName("Dense_b"), MatMul(scope.WithOpName("Dense_w"), std::move(input), m_vars["W"+idx]), m_vars["B"+idx]);
    if(bActivation)
        return Relu(scope.WithOpName("Relu"), dense);
    else
        return dense;
}

Status LayersStackBuilder::CreateGraphForCNN(int filter_side)
{
    //input image is batch_sizex150x150x3
    input_batch_var = Placeholder(t_root.WithOpName(input_name), DT_FLOAT);
    drop_rate_var = Placeholder(t_root.WithOpName(drop_rate_name), DT_FLOAT);//see class member for help
    skip_drop_var = Placeholder(t_root.WithOpName(skip_drop_name), DT_FLOAT);//see class member for help

    //Start Conv+Maxpool No 1. filter size 3x3x3 and we have 32 filters
    Scope scope_conv1 = t_root.NewSubScope("Conv1_layer");
    int in_channels = image_channels;
    int out_channels = 32;
    auto pool1 = AddConvLayer("1", scope_conv1, in_channels, out_channels, filter_side, input_batch_var);
    int new_side = ceil((float)image_side / 2); //max pool is reducing the size by factor of 2

    //Conv+Maxpool No 2
    Scope scope_conv2 = t_root.NewSubScope("Conv2_layer");
    in_channels = out_channels;
    out_channels = 64;
    auto pool2 = AddConvLayer("2", scope_conv2, in_channels, out_channels, filter_side, pool1);
    new_side = ceil((float)new_side / 2);

    //Conv+Maxpool No 3
    Scope scope_conv3 = t_root.NewSubScope("Conv3_layer");
    in_channels = out_channels;
    out_channels = 128;
    auto pool3 = AddConvLayer("3", scope_conv3, in_channels, out_channels, filter_side, pool2);
    new_side = ceil((float)new_side / 2);

    //Conv+Maxpool No 4
    Scope scope_conv4 = t_root.NewSubScope("Conv4_layer");
    in_channels = out_channels;
    out_channels = 128;
    auto pool4 = AddConvLayer("4", scope_conv4, in_channels, out_channels, filter_side, pool3);
    new_side = ceil((float)new_side / 2);

    //Flatten
    Scope flatten = t_root.NewSubScope("flat_layer");
    int flat_len = new_side * new_side * out_channels;
    auto flat = Reshape(flatten, pool4, {-1, flat_len});

    //Dropout
    Scope dropout = t_root.NewSubScope("Dropout_layer");
    auto rand = RandomUniform(dropout, Shape(dropout, flat), DT_FLOAT);
    //binary = floor(rand + (1 - drop_rate) + skip_drop)
    auto binary = Floor(dropout, Add(dropout, rand, Add(dropout, Sub(dropout, 1.f, drop_rate_var), skip_drop_var)));
    auto after_drop = Multiply(dropout.WithOpName("dropout"), Div(dropout, flat, drop_rate_var), binary);

    //Dense No 1
    int in_units = flat_len;
    int out_units = 512;
    Scope scope_dense1 = t_root.NewSubScope("Dense1_layer");
    auto relu5 = AddDenseLayer("5", scope_dense1, in_units, out_units, true, after_drop);

    //Dense No 2
    in_units = out_units;
    out_units = 256;
    Scope scope_dense2 = t_root.NewSubScope("Dense2_layer");
    auto relu6 = AddDenseLayer("6", scope_dense2, in_units, out_units, true, relu5);

    //Dense No 3
    in_units = out_units;
    out_units = 1;
    Scope scope_dense3 = t_root.NewSubScope("Dense3_layer");
    auto logit_layer = AddDenseLayer("7", scope_dense3, in_units, out_units, false, relu6);

    out_classification = Sigmoid(t_root.WithOpName(out_name), logit_layer);
    return t_root.status();
}

Status LayersStackBuilder::CreateOptimizationGraph(float learning_rate)
{
    input_labels_var = Placeholder(t_root.WithOpName("inputL"), DT_FLOAT);
    Scope scope_loss = t_root.NewSubScope("Loss_scope");
    out_loss_var = Mean(scope_loss.WithOpName("Loss"), SquaredDifference(scope_loss, out_classification, input_labels_var), {0});
    TF_CHECK_OK(scope_loss.status());
    for(const auto& i: m_vars)
        v_weights_biases.push_back(i.second);
    vector<Output> grad_outputs;
    TF_CHECK_OK(AddSymbolicGradients(t_root, {out_loss_var}, v_weights_biases, &grad_outputs));
    int index = 0;
    for(const auto& i: m_vars)
    {
        //Applying Adam
        string s_index = to_string(index);
        auto m_var = Variable(t_root, m_shapes[i.first], DT_FLOAT);
        auto v_var = Variable(t_root, m_shapes[i.first], DT_FLOAT);
        m_assigns["m_assign"+s_index] = Assign(t_root, m_var, Input::Initializer(0.f, m_shapes[i.first]));
        m_assigns["v_assign"+s_index] = Assign(t_root, v_var, Input::Initializer(0.f, m_shapes[i.first]));

        auto adam = ApplyAdam(t_root, i.second, m_var, v_var, 0.f, 0.f, learning_rate, 0.9f, 0.999f, 0.00000001f, {grad_outputs[index]});
        v_out_grads.push_back(adam.operation);
        index++;
    }
    return t_root.status();
}

Status LayersStackBuilder::Initialize()
{
    if(!t_root.ok())
        return t_root.status();

    vector<Output> ops_to_run;
    for(const auto& i: m_assigns)
        ops_to_run.push_back(i.second);
    t_session = std::make_unique<ClientSession>(t_root);
    TF_CHECK_OK(t_session->Run(ops_to_run, nullptr));
    /* uncomment if you want visualization of the model graph
    GraphDef graph;
    TF_RETURN_IF_ERROR(t_root.ToGraphDef(&graph));
    SummaryWriterInterface* w;
    TF_CHECK_OK(CreateSummaryFileWriter(1, 0, "/Users/bennyfriedman/Code/TF2example/TF2example/graphs", ".cnn-graph", Env::Default(), &w));
    TF_CHECK_OK(w->WriteGraph(0, make_unique<GraphDef>(graph)));
    */
    return Status::OK();
}

Status LayersStackBuilder::TrainCNN(Tensor& image_batch, Tensor& label_batch, vector<float>& results, float& loss)
{
    if(!t_root.ok())
        return t_root.status();

    vector<Tensor> out_tensors;
    //Inputs: batch of images, labels, drop rate and do not skip drop.
    //Extract: Loss and result. Run also: Apply Adam commands
    TF_CHECK_OK(t_session->Run({{input_batch_var, image_batch}, {input_labels_var, label_batch}, {drop_rate_var, 0.5f}, {skip_drop_var, 0.f}}, {out_loss_var, out_classification}, v_out_grads, &out_tensors));
    loss = out_tensors[0].scalar<float>()(0);
    //both labels and results are shaped [20, 1]
    auto mat1 = label_batch.matrix<float>();
    auto mat2 = out_tensors[1].matrix<float>();
    for(int i = 0; i < mat1.dimension(0); i++)
        results.push_back((fabs(mat2(i, 0) - mat1(i, 0)) > 0.5f)? 0 : 1);
    return Status::OK();
}

Status LayersStackBuilder::ValidateCNN(Tensor& image_batch, Tensor& label_batch, vector<float>& results)
{
    if(!t_root.ok())
        return t_root.status();

    vector<Tensor> out_tensors;
    //Inputs: batch of images, drop rate 1 and skip drop.
    TF_CHECK_OK(t_session->Run({{input_batch_var, image_batch}, {drop_rate_var, 1.f}, {skip_drop_var, 1.f}}, {out_classification}, &out_tensors));
    auto mat1 = label_batch.matrix<float>();
    auto mat2 = out_tensors[0].matrix<float>();
    for(int i = 0; i < mat1.dimension(0); i++)
        results.push_back((fabs(mat2(i, 0) - mat1(i, 0)) > 0.5f)? 0 : 1);
    return Status::OK();
}

Status LayersStackBuilder::Predict(Tensor& image, int& result)
{
    if(!t_root.ok())
        return t_root.status();

    vector<Tensor> out_tensors;
    //Inputs: image, drop rate 1 and skip drop.
    TF_CHECK_OK(t_session->Run({{input_batch_var, image}, {drop_rate_var, 1.f}, {skip_drop_var, 1.f}}, {out_classification}, &out_tensors));
    auto mat = out_tensors[0].matrix<float>();
    result = (mat(0, 0) > 0.5f)? 1 : 0;
    return Status::OK();
}

Status LayersStackBuilder::FreezeSave(string& file_name)
{
    vector<Tensor> out_tensors;
    //Extract: current weights and biases current values
    TF_CHECK_OK(t_session->Run(v_weights_biases , &out_tensors));
    GraphDef graph_def;
    TF_CHECK_OK(t_root.ToGraphDef(&graph_def));
    //call the utility function (modified)
    SavedModelBundle saved_model_bundle;
    SignatureDef signature_def;
    (*signature_def.mutable_inputs())[input_batch_var.name()].set_name(input_batch_var.name());
    (*signature_def.mutable_outputs())[out_classification.name()].set_name(out_classification.name());
    MetaGraphDef* meta_graph_def = &saved_model_bundle.meta_graph_def;
    (*meta_graph_def->mutable_signature_def())["signature_def"] = signature_def;
    *meta_graph_def->mutable_graph_def() = graph_def;
    SessionOptions session_options;
    saved_model_bundle.session.reset(NewSession(session_options));//even though we will not use it
    GraphDef frozen_graph_def;
    std::unordered_set<string> inputs;
    std::unordered_set<string> outputs;
    TF_CHECK_OK(tensorflow::FreezeSavedModel(saved_model_bundle, &frozen_graph_def, &inputs, &outputs));

    //write to file
    return WriteBinaryProto(Env::Default(), file_name, frozen_graph_def);
}

Status LayersStackBuilder::LoadSavedModel(string& file_name)
{
    std::unique_ptr<GraphDef> graph_def;
    SessionOptions options;
    f_session.reset(NewSession(options));
    graph_def = std::make_unique<GraphDef>();
    TF_CHECK_OK(ReadBinaryProto(Env::Default(), file_name, graph_def.get()));
    return f_session->Create(*graph_def);
}

Status LayersStackBuilder::PredictFromFrozen(Tensor& image, int& result)
{
    vector<Tensor> out_tensors;
    Tensor t(DT_FLOAT, TensorShape({1}));
    t.scalar<float>()(0) = 1.f;
    //Inputs: image, drop rate 1 and skip drop.
    TF_CHECK_OK(f_session->Run({{input_name, image}, {drop_rate_name, t}, {skip_drop_name, t}}, {out_name}, {}, &out_tensors));
    auto mat = out_tensors[0].matrix<float>();
    result = (mat(0, 0) > 0.5f)? 1 : 0;
    return Status::OK();
}

Status LayersStackBuilder::WriteBatchToImageFiles(Tensor& image_batch, const string& folder_name, const string& image_name)
{
    Scope root = Scope::NewRootScope();
    auto unormalized = Multiply(root, image_batch, 255.f);
    auto cast = Cast(root, unormalized, DT_UINT8);
    auto image_list = Unstack(root, cast, image_batch.dim_size(0));
    size_t num = image_list.output.size();
    vector<Output> images;
    for(int i = 0; i < num; i++)
        images.push_back(EncodeJpeg(root, image_list.output[i]));

    TF_CHECK_OK(root.status());
    ClientSession session(root);
    vector<Tensor> out_tensors;
    TF_CHECK_OK(session.Run({}, {images}, &out_tensors));

    for(int i = 0; i < num; i++)
    {
        string i_name = image_name + to_string(i) + ".jpg";
        string i_fullpath = io::JoinPath(folder_name, i_name);
        ofstream fs(i_fullpath, ios::binary);
        fs << out_tensors[i].scalar<tstring>()();
    }
    return Status::OK();
}
