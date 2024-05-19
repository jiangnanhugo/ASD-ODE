#!/usr/bin/zsh
basepath=/home/$USER/PycharmProjects/act_ode
py3=/home/$USER/miniconda3/bin/python
# Glycolytic_oscillator_d0
type=Strogatz
opt=Nelder-Mead
noise_type=normal
noise_scale=0.0
metric_name=neg_mse
n_cores=10
num_init_conds=5
nvars=$1
total_progs=$2
set -x
for width in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    for num_regions in 5 10 15 20 25 30 35 40 45 50
    do
for ei in {1..${total_progs}}; do
	eq_name=${nvars}_prog${ei}
	echo "submit $eq_name"

	dump_dir=$basepath/result/${type}/$(date +%F)
	echo $dump_dir
	if [ ! -d "$dump_dir" ]; then
		echo "create output dir: $dump_dir"
		mkdir -p $dump_dir
	fi
	CUDA_VISIBLE_DEVICES="" $py3 $basepath/ablation_study/benchmark_active_learning.py  --equation_name $eq_name \
		--optimizer $opt --metric_name $metric_name --num_init_conds $num_init_conds --noise_type $noise_type --noise_scale $noise_scale --n_cores $n_cores >$dump_dir/Eq_${eq_name}.noise_${noise_type}${noise_scale}.opt$opt.act_dso.out
done
done
done