## Overview

This repo provides code for analyzing satellite swarms. Most of the files are just dummy examples. The most recent iteration is `nl_recursive_combined_state.py` which implements a combined landmark, inter-range measurement model and uses this to solve the problem in a recursive fashion. In addition it calculates the FIM and finally plots the combined variance and CRB. 

`nl_recursive_mc_live.py` provides live, continuous error plotting capabilities.

`nl_least_squares_monte_carlo.py` does something similar in a least squares approach as v2. It currently does not have any plotting capabilities. 

Only v2 and live keep getting developed so any other file does not necessarily have all the features/bug-fixes of these two.

Run the following command to setup up the environment:

```
conda create --name my_env python=3.12 -y
```

```
conda activate my_env
```

```
conda install -c conda-forge ipopt -y
```

```
pip install -r requirements.txt
```


