#!/bin/sh

if [ "$1" = "--irq" ]; then
echo "Setting affinities for PCI IRQs."
echo 03f > /proc/irq/24/smp_affinity
echo 03f > /proc/irq/30/smp_affinity
echo fc0 > /proc/irq/48/smp_affinity
echo fc0 > /proc/irq/54/smp_affinity
elif [ "$1" = "--msi" ]; then
echo "Setting affinities for MSI IRQs. (NUMA Clusters)"
echo 03f > /proc/irq/100/smp_affinity
echo 03f > /proc/irq/101/smp_affinity
echo 03f > /proc/irq/102/smp_affinity
echo 03f > /proc/irq/103/smp_affinity
echo fc0 > /proc/irq/104/smp_affinity
echo fc0 > /proc/irq/105/smp_affinity
echo fc0 > /proc/irq/106/smp_affinity
echo fc0 > /proc/irq/107/smp_affinity
elif [ "$1" = "--msi-partitioned" ]; then
echo "Setting affinities for MSI IRQs (Partitioned)."
echo 001 > /proc/irq/100/smp_affinity
echo 002 > /proc/irq/101/smp_affinity
echo 004 > /proc/irq/102/smp_affinity
echo 008 > /proc/irq/103/smp_affinity
echo 040 > /proc/irq/104/smp_affinity
echo 080 > /proc/irq/105/smp_affinity
echo 100 > /proc/irq/106/smp_affinity
echo 200 > /proc/irq/107/smp_affinity
elif [ "$1" = "--msi-small-cluster" ]; then
echo "Setting affinities for MSI IRQs (Two clusters per NUMA node)."
echo 007 > /proc/irq/100/smp_affinity
echo 007 > /proc/irq/101/smp_affinity
echo 038 > /proc/irq/102/smp_affinity
echo 038 > /proc/irq/103/smp_affinity
echo 1c0 > /proc/irq/104/smp_affinity
echo 1c0 > /proc/irq/105/smp_affinity
echo e00 > /proc/irq/106/smp_affinity
echo e00 > /proc/irq/107/smp_affinity
elif [ "$1" = "--msi-prt" ]; then
echo "Setting affinities for MSI IRQs (NUMA Clusters, PREEMPT_RT)."
echo 03f > /proc/irq/112/smp_affinity
echo 03f > /proc/irq/113/smp_affinity
echo 03f > /proc/irq/114/smp_affinity
echo 03f > /proc/irq/115/smp_affinity
echo fc0 > /proc/irq/116/smp_affinity
echo fc0 > /proc/irq/117/smp_affinity
echo fc0 > /proc/irq/118/smp_affinity
echo fc0 > /proc/irq/119/smp_affinity
else
echo "Unknown option."
exit 1
fi

#/usr/sbin/irqbalance --oneshot
