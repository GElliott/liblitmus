#!/bin/bash

nvidia-smi -pm 1
./nvidia_numa_irq.sh --msi

echo L3 > /proc/litmus/plugins/C-EDF/cluster
echo C-EDF > /proc/litmus/active_plugin
