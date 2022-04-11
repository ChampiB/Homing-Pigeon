# Format of the files with extension .lake

This document specify the format that a lake file must have to be loaded properly by
the MazeEnv class.

The first row of the file must contains to integer describing the number of rows
and columns in the lake, e.g.

```
7 8
```

indicates that the lake contains in this file contains 7 rows and 8 columns. The
first line must be followed by n other lines where n is the number of rows, i.e.
each line after the first line specify the content of one row in the lake. For
example:

```
WWWWWWWW
W.....EW
W.WWWW.W
W....W.W
W.WW.W.W
WS.....W
WWWWWWWW
```

Various characters can be used to specify a lake's row:
- 'W' stands for wall, i.e. the agent cannot pass through it;
- '.' refers to empty cell, i.e. the agent can navigate them;
- 'E' specify where the lake exit is, i.e. agent's target location;
- 'S' specify the initial position of the agent in the lake.

Importantly, the lake is assumed to be surrounded by walls!

Any variation from this format might lead to crashes, bugs or exceptions threw.