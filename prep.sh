#!/bin/bash

for i in $(ps -ef | grep -i rcu | grep -v grep | awk '{print $2}'); do taskset -p 0x00000FC0 $i; done
for i in $(ps -ef | grep -i irq/ | grep -v grep | awk '{print $2}'); do taskset -p 0x00000FC0 $i; done
for i in $(ps -ef | grep -i khugepaged | grep -v grep | awk '{print $2}'); do taskset -p 0x00000FC0 $i; done
nvidia-smi -pm 1
./nvidia_numa_irq.sh --msi
service mysql stop
service smbd stop
echo L3 > /proc/litmus/plugins/C-EDF/cluster
./setsched C-EDF

