#!/usr/bin/zsh
basepath=/home/$USER/data/act_ode
py3=/home/$USER/workspace/miniconda3/envs/py310/bin/python
#
type=Strogatz
opt=Nelder-Mead
noise_type=normal
noise_scale=0.0
metric_name=neg_mse
n_cores=8
num_init_conds=5
nvars=$1
total_progs=$2
for ei in {1..${total_progs}};
do
  eq_name=${nvars}_prog${ei}
	echo "submit $eq_name"
  log_dir=$basepath/log/$(date +%F)
	echo $log_dir
	if [ ! -d "$log_dir" ]; then
		echo "create dir: $log_dir"
		mkdir -p $log_dir
	fi
	dump_dir=$basepath/result/${type}/$(date +%F)
	echo $dump_dir
	if [ ! -d "$dump_dir" ]; then
		echo "create output dir: $dump_dir"
		mkdir -p $dump_dir
	fi
	sbatch -A yexiang --nodes=1 --ntasks=1 --cpus-per-task=${n_cores} <<EOT
#!/bin/bash -l

#SBATCH --job-name="ODE-V${nvars}-Pg${total_progs}"
#SBATCH --output=$log_dir/${eq_name}.noise_${noise_type}_${noise_scale}.opt$opt.act_dso.out
#SBATCH --constraint=A
#SBATCH --time=12:00:00
#SBATCH --mem=8GB

hostname

$py3 $basepath/act_dso/main.py $basepath/act_dso/config/config_regression.json --equation_name $eq_name \
		--optimizer $opt --metric_name $metric_name --num_init_conds $num_init_conds --noise_type $noise_type --noise_scale $noise_scale  --n_cores $n_cores  >$dump_dir/Eq_${eq_name}.noise_${noise_type}${noise_scale}.opt$opt.act_dso.out &

EOT
done
