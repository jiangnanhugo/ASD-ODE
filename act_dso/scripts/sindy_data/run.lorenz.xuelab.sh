#!/usr/bin/zsh
basepath=/home/$USER/data/act_ode
py3=/home/$USER/miniconda3/envs/py310/bin/python3.10
#
type=Lorenz
#datapath=$basepath/data/differential_equations/
opt=L-BFGS-B
noise_type=normal
noise_scale=0.0
metric_name=inv_mse
n_cores=1
set -x
for eq_name in Lorenz;
do
  echo "submit $eq_name"

	dump_dir=$basepath/result/${type}/$(date +%F)
	echo $dump_dir
	if [ ! -d "$dump_dir" ]; then
		echo "create output dir: $dump_dir"
		mkdir -p $dump_dir
	fi
	for bsl in DSR; do
		echo $basepath/cvDSO/config/config_regression_${bsl}.json
		CUDA_VISIBLE_DEVICES=""  $py3 $basepath/act_dso/main.py $basepath/act_dso/config/config_regression_${bsl}.json --equation_name $eq_name \
			--optimizer $opt --metric_name $metric_name --n_cores $n_cores --noise_type $noise_type --noise_scale $noise_scale #>$dump_dir/Eq_${eq_name}.noise_${noise_type}${noise_scale}.opt$opt.${bsl}.cvdso.out &
	done
done
