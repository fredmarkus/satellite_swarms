## Overview

This repo provides code for analyzing satellite swarms. Most of the files are just dummy examples. The most recent iteration is `nl_recursive_combined_state.py` which implements a combined landmark, inter-range measurement model and uses this to solve the problem in a recursive fashion. In addition it calculates the FIM and finally plots the combined variance and CRB. 

`nl_recursive_mc_live.py` provides live, continuous error plotting capabilities.

Run the following command to setup up the environment:

```
conda env create -f environment.yml --name sat_env
```


