#!/usr/bin/zsh
basepath=/home/$USER/data/cvdso
py3=/home/$USER/workspace/miniconda3/envs/py310/bin/python

type=Glycolytic_oscillator
datapath=$basepath/data/algebraic_equations/differential_equations
opt=L-BFGS-B
noise_type=normal
noise_scale=0.0
metric_name=inv_nrmse

for prog in {0..6};
do
	eq_name=${type}_d${prog}.in
    echo "submit $eq_name"
    
    dump_dir=$basepath/result/${type}/$(date +%F)
	echo $dump_dir
	if [ ! -d "$dump_dir" ]; then
		echo "create output dir: $dump_dir"
		mkdir -p $dump_dir
	fi
	log_dir=$basepath/log/$(date +%F)
	echo $log_dir
	if [ ! -d "$log_dir" ]; then
		echo "create dir: $log_dir"
		mkdir -p $log_dir
	fi
	for bsl in DSR; do
		echo $basepath/cvDSO/config/config_regression_${bsl}.json
		sbatch -A yexiang --nodes=1 --ntasks=1 --cpus-per-task=1 <<EOT
#!/bin/bash -l

#SBATCH --job-name="cvDSO-${type}_d${prog}"
#SBATCH --output=$log_dir/${eq_name}.noise_${noise_type}_${noise_scale}.${bsl}.cvdso.out
#SBATCH --constraint=A
#SBATCH --time=48:00:00
#SBATCH --mem=8GB

hostname

$py3 $basepath/cvDSO/main.py $basepath/cvDSO/config/config_regression_${bsl}.json --equation_name $datapath/$eq_name \
--optimizer $opt --metric_name $metric_name \
--noise_type $noise_type --noise_scale $noise_scale  >  $dump_dir/prog_${prog}.noise_${noise_type}${noise_scale}.opt$opt.${bsl}.cvdso.out

EOT
	done
done

#done
