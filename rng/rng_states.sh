#!/bin/bash -l 

name=rng_states

nvcc $name.cu -ccbin /usr/bin/clang  -o /tmp/$name && /tmp/$name && python -c "import numpy as np ; print(np.load('/tmp/rng.npy'))"

