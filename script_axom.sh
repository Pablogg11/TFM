#!/bin/bash

cat script.sh

output_path="results/classification/axom"

mkdir -p "${output_path}/csv"

for dataset in "gaussianPiecewiseConstant", "gaussianLinear", gaussianNonLinearAdditive" ;
do
    echo "Running experiment for ${dataset}"
    for rho in 0.5
    do
        python main_driver.py --mode classification --seed 7 --experiment --experiment-json configs/experiment_config_axom.jsonc --rho $rho --dataset $dataset --results-dir $output_path &
    done
    wait
done
wait