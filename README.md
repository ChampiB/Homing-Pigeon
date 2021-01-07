![HoPi logo](hopi-logo.png)

# Homing-Pigeon (Version 1.0)

----------------------
Homing Pigeon (hopi) provides a modular implementation of variational message passing in C++. Thus, it provides an 
easy to use interface to create generative model using Forney factor graph. Currently, three types of factors are 
available:
- Categorical, i.e P(s)
- Transition, i.e. P(o|s)
- ActiveTransition, i.e. P(s1|so,action)

The third node allows the user to implement active inference agents and agents using planning as inference. Finally, the
current version also allows planning using tree search, where three search here refers to dynamic expansions of the 
generative model. The tree expansion is driven by a heuristic (expected free energy) and the action selection is driven 
by the number of times each root's child has been expanded (ties are broken based on the heuristic).

## Documentation

----------------------

The Homing Pigeon documentation is available on the [github wiki](https://github.com/ChampiB/Homing-Pigeon/wiki).

## Issues for reporting bugs

----------------------
Even if we maintain the project is properly tested using the unit test library catch, bugs may pass through. If you 
discover one, please open an issue on the project repository. Note, that the issue's name must be of the following 
format:

```[BUG] <Short Explanation Of The Bug>```

We are doing our best to provide quick fix of current bugs. If you cannot afford to wait a few days/weeks feel free to 
contribute to the bug resolution.

## Issues for asking new features

----------------------
At any time, new features can be requested by simply opening an issue. The issue's name must be formatted as follows:

```[Feature] <Short Explanation Of The Feature>```

We are doing our best to implement the requested features, but we cannot guarantee that all of them will be implemented
fast, feel free to contribute to the project if you want a feature to be integrated quickly.

## Contributions

----------------------
Since this project is open source, we encourage external contributions. The rule is simple: "To contribute create a new 
branch related to an issue (bug fixing or new feature), and when the code is ready to be integrated to the project core 
create a merge request". Note that test coverage is mandatory for the acceptance of any merge request (no test, no 
merge).

## Typos and grammatical mistakes 

----------------------

If you spot any typos or grammatical mistakes, please contact: ```tmac3@kent.ac.uk```.
