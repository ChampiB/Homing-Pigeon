![HoPi logo](hopi-logo.png)

# Homing-Pigeon (Version 1.0)

----------------------
Homing Pigeon (hopi) provides a modular implementation of variational message passing in C++. Thus, it provides an 
easy-to-use interface to create generative model using Forney factor graph. Currently, four types of factors are 
available:
- Categorical, i.e, P(s) = Cat(param)
- Transition, i.e., P(o|s) = Cat(param)
- ActiveTransition, i.e., P(s1|s0,action) = Cat(param)
- Dirichlet, i.e., P(A) = Dir(param)

The third node allows the user to implement active inference agents that perform planning as inference. Finally, the
current version also allows planning using tree search, where three search here refers to dynamic expansion of the 
generative model. The expected free energy (EFE) drives the tree expansion, and the number of times each root's child
has been expanded drives action selection (ties are broken based on the EFE).

## Build

----------------------

Please refer to the following file to build Homing-Pigeon: `BUILD.md`.

## Documentation

----------------------

The Homing Pigeon documentation is available on the [github wiki](https://github.com/ChampiB/Homing-Pigeon/wiki).

## Issues for reporting bugs

----------------------
Even if the project is properly tested using the unit test library catch, bugs may pass through. If you 
discover one, please open an issue on the project repository. Note, that the issue's name must be of the following 
format:

> [BUG] Short Explanation Of The Bug

We are doing our best to provide quick fix of current bugs. If you cannot afford to wait a few days/weeks feel free to 
contribute to the bug resolution.

## Issues for asking new features

----------------------
At any time, new features can be requested by simply opening an issue. The issue's name must be formatted as follows:

> [Feature] Short Explanation Of The Feature

We are doing our best to implement the requested features, but we cannot guarantee that all of them will be implemented
fast. Feel free to contribute to the project if you want a feature to be integrated quickly.

## Contributions

----------------------
Since this project is open source, we encourage external contributions. The rule is simple: "To contribute create a new 
branch from "dev" related to an issue (bug fixing or new feature). Implement the changes, and create a merge request".
Note that test coverage is mandatory for the acceptance of any merge request (no test, no merge).

## Typos and grammatical mistakes 

----------------------

If you spot any typos or grammatical mistakes, please contact: ```tmac3@kent.ac.uk```.
