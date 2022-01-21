# Building Homing-Pigeon

----------------------

Homing-Pigeon is using cmake for the build. First you need to install Pytorch C++ API in the directory `libs/torch`:
- `cd libs`
- `wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip`
- `unzip libtorch-shared-with-deps-latest.zip`
- `mv libtorch torch`
- `rm libtorch-shared-with-deps-latest.zip`
- `cd ..`

Next, if boost is not installed on your system, run the following command:
- `sudo apt-get install libboost-all-dev`

Similarly, if gnuplot is not installed on your system, run the following command:
- `sudo apt-get install -y gnuplot libgnuplot-iostream-dev`

Then, use cmake to build the project as follows:
- `mkdir build`
- `cd build`
- `cmake ..`
- `make`
- `cd ..`

The above set of commands will build the Homing-Pigeon library, the unit tests as well as the examples of the project. Note that the example named `deep_learning_mnist` requires the mnist dataset to be present in the build directory. You can download the dataset using the following command:
- `mkdir build/mnist`
- `cd build/mnist`
- `git clone https://github.com/HIPS/hypergrad.git`
- `mv hypergrad/data/mnist/* .`
- `rm -r hypergrad mnist_data.pkl`
- `gunzip t*-ubyte.gz`
- `cd ../..`
