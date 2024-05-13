#!/bin/bash -l

basepath=/home/jiangnan/PycharmProjects/scibench
type=Livermore2
nv=$1

thispath=$basepath/mcts_and_vsr_mcts
data_path=$basepath/data/unencrypted/equations_livermore2
py3=/home/jiangnan/miniconda3/bin/python
opt=L-BFGS-B

noise_type=normal
noise_scale=0.0
metric_name=neg_mse
for prog in {1..10}; do
	eq_name=${type}_Vars${nv}_$prog.in
	echo "submit $eq_name"
	dump_dir=$basepath/result/${type}_Vars${nv}/$(date +%F)
	if [ ! -d "$dump_dir" ]; then
		echo "create dir: $dump_dir"
		mkdir -p $dump_dir
	fi
	log_dir=$basepath/log/$(date +%F)
	if [ ! -d "$log_dir" ]; then
		echo "create dir: $log_dir"
		mkdir -p $log_dir
	fi
	echo $dump_dir/prog_${prog}.metric_${metric_name}.noise_${noise_type}${noise_scale}.opt$opt.cv_mcts.out
	nohup $py3 $thispath/main.py --equation_name $data_path/$eq_name --optimizer $opt --cv_mcts --metric_name $metric_name \
		--production_rule_mode 'livermore2' \
		--noise_type $noise_type --noise_scale $noise_scale >$dump_dir/prog_${prog}.metric_${metric_name}.noise_${noise_type}${noise_scale}.opt$opt.cv_mcts.out &

done
