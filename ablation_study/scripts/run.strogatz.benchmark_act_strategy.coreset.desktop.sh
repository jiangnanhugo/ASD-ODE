#!/usr/bin/zsh
basepath=/home/$USER/PycharmProjects/act_ode
py3=/home/$USER/miniconda3/bin/python
# Glycolytic_oscillator_d0
type=Strogatz
noise_type=normal
noise_scale=0.0
active_mode=coreset
num_init_conds=100
nvars=vars2
set -x

ei=10
eq_name=${nvars}_prog${ei}
	echo "submit $eq_name"

	dump_dir=$basepath/result/${type}/$(date +%F)
	echo $dump_dir
	if [ ! -d "$dump_dir" ]; then
		echo "create output dir: $dump_dir"
		mkdir -p $dump_dir
	fi
$py3 $basepath/ablation_study/benchmark_active_learning_strategies.py  --equation_name $eq_name \
	--track_memory True --memray_output_bin $active_mode.output.bin \
	--active_mode coreset \
	--pred_expressions_file /home/jiangnan/PycharmProjects/act_ode/ablation_study/pred_expressions/vars2_prog10.out \
--num_init_conds $num_init_conds --noise_type $noise_type --noise_scale $noise_scale #>$dump_dir/Eq_${eq_name}.noise_${noise_type}${noise_scale}.act.coreset.out
