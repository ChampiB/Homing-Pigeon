# Format of the files with extension .evi

This document specify the format that an evidence file must have to be loaded properly by
the FactorGraph::loadEvidence function.

Each file's row contains two parts. The first is the variable's name and the second is an integer corresponding to the observation made, e.g.

```
o0 2
o1 1
o2 0
o3 1
```

When this file is loaded the integer is turned into a one hot vector based on the number of observations in the environment, e.g.

```
if env->observations() == 5
then "o1 1" --> "[0 1 0 0 0]"
```

Importantly, the name is not allowed to contain spaces and the evidence can only be loaded in observed variables.

Any variation from this format might lead to crashes, bugs or exceptions threw.