## Overview

This repo provides code for analyzing satellite swarms. Most of the files are just dummy examples. The most recent iteration is 'nl_least_squares_jax_adv.py' which implements a combined landmark, inter-range measurement model and uses this to solve the nonlinear least squares problem using the cyipopt solver. 


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


