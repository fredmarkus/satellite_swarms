## Overview

This repo provides code for analyzing satellite swarms. Most of the files are just dummy examples. The most recent iteration is `nl_recursive_monte_carlo_v2.py` which implements a combined landmark, inter-range measurement model and uses this to solve the problem in a recursive fashion. In addition it calculates the FIM and finally plots the combined variance and CRB. 

`nl_least_squares_monte_carlo.py` does something similar in a least squares approach. It currently does not have any plotting capabilities. 


## Requirements: 

To run the code, the following packages are needed: 
- jax
- cyipopt
- numpy
- yaml

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


