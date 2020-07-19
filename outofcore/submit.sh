#!/bin/bash

bsub -q debug -alloc_flags gpudefault -W 00:30 -nnodes 1 -P CSC362 -env "all,LSF_CPU_ISOLATION=on" -Is /bin/bash
