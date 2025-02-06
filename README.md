## Overview

This repo provides code for analyzing satellite swarms. Most of the files are just dummy examples. The most recent iteration is `nl_recursive_combined_state.py` which implements a combined landmark, inter-range, relative-bearing measurement model and uses this to solve the problem in a recursive fashion. In addition it calculates the FIM and finally plots the combined variance and CRB as well as a bunch of other stuff.

## Setup
Run the following command to setup up the environment:

```
conda env create -f environment.yml --name sat_env
```

## Run
The file has the following arguments: 

`--N` determines the number of timesteps

`--f` update frequency

`--ignore_earth` whether the satellites should consider earth when taking relative measurements

`--num_trials` number of Monte Carlo trials

`--random_yaml` whether to use random satellite configurations to run

`--run_all` when running with multiple satellites this will run the sim for every single number up to the specified number of satellites

`--state_dim` number of states per satellite (default 6 and nothing else is supported)

`--verbose` print logger measurements and timesteps

`--measurement_type` the type of measurement to use. options are 'land', 'range', 'sat_bearing'. Specify as: `--measurement_type land range` for example
