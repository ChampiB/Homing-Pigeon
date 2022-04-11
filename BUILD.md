# Building Homing-Pigeon

----------------------

Homing-Pigeon is using cmake for the build. First you need to install Pytorch C++ API in the directory `libs/torch`:
- `cd libs`
- `wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.10.2%2Bcpu.zip`
- `unzip libtorch-cxx11-abi-shared-with-deps-1.10.2%2Bcpu.zip`
- `mv libtorch torch`
- `rm libtorch-cxx11-abi-shared-with-deps-1.10.2%2Bcpu.zip`
- `cd ..`

Second, install OpenCV:
- `sudo apt-get install libopencv-dev`

Then, use cmake to build the project as follows:
- `mkdir build`
- `cd build`
- `cmake ..`
- `make`
- `cd ..`

Next, if you want to run the simulation involving the dSprites dataset run the following commands:
- `cd examples/d_sprites`
- `git clone https://github.com/deepmind/dsprites-dataset.git`
- `python3 ./d_sprites_from_npz_to_pickle.py`
- `cd ../..`

The above set of commands will build the Homing-Pigeon library, the unit tests as well as the examples of the project. Note that the example named `deep_learning_mnist` requires the mnist dataset to be present in the build directory. You can download the dataset using the following command:
- `mkdir build/mnist`
- `cd build/mnist`
- `git clone https://github.com/HIPS/hypergrad.git`
- `mv hypergrad/data/mnist/* .`
- `rm -r hypergrad mnist_data.pkl`
- `gunzip t*-ubyte.gz`
- `cd ../..`
