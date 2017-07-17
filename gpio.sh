#!/bin/bash
#odroid XU4.
clear
echo 33 > /sys/class/gpio/export
echo "out" > /sys/class/gpio/gpio33/direction

