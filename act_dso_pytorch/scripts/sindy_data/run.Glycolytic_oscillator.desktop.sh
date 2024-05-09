#!/usr/bin/zsh
basepath=/home/$USER/PycharmProjects/cvdso
py3=/home/$USER/miniconda3/bin/python
# Glycolytic_oscillator_d0
type=Glycolytic_oscillator
datapath=$basepath/data/differential_equations/
opt=L-BFGS-B
noise_type=normal
noise_scale=0.0
metric_name=inv_mse
n_cores=1
set -x
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
	for bsl in DSR; do
		echo $basepath/cvDSO/config/config_regression_${bsl}.json
		CUDA_VISIBLE_DEVICES="" nohup $py3 $basepath/cvDSO/main.py $basepath/cvDSO/config/config_regression_${bsl}.json --equation_name $datapath/$eq_name \
			--optimizer $opt --metric_name $metric_name --n_cores $n_cores --noise_type $noise_type --noise_scale $noise_scale >$dump_dir/prog_${prog}.noise_${noise_type}${noise_scale}.opt$opt.${bsl}.cvdso.out &
	done
done
