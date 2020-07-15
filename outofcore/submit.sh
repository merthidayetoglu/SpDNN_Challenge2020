#!/bin/bash

bsub -q debug -alloc_flags gpudefault -W 01:00 -nnodes 1 -P CSC362 -env "all,LSF_CPU_ISOLATION=on" -Is /bin/bash
