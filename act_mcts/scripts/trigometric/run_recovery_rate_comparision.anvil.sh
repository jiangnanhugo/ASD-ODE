#!/usr/bin/zsh
set -x
basepath=/anvil/projects/x-cis230379/jiang631/data/scibench
py310=/home/x-jiang631/workspace/miniconda3/envs/py310/bin/python3

scatch_basepath=/anvil/scratch/x-jiang631/data/scibench
type=$1
nv=$2
nt=$3

thispath=$basepath/mcts_and_vsr_mcts
data_path=$basepath/data/unencrypted/equations_trigometric
opt=L-BFGS-B

noise_type=normal
noise_scale=0.0
metric_name=neg_mse
for prog in {0..9}; do
	eq_name=${type}_nv${nv}_nt${nt}_prog_${prog}.in
	echo "submit $eq_name"
	dump_dir=$basepath/result/${type}_nv${nv}_nt${nt}/$(date +%F)
	if [ ! -d "$dump_dir" ]; then
		echo "create dir: $dump_dir"
		mkdir -p $dump_dir
	fi
	scratch_dir=$scatch_basepath/result/${type}_nv${nv}_nt${nt}/$(date +%F)
	if [ ! -d "$scratch_dir" ]; then
		echo "create dir: $scratch_dir"
		mkdir -p $scratch_dir
	fi
	log_dir=$basepath/log/$(date +%F)
	if [ ! -d "$log_dir" ]; then
		echo "create dir: $log_dir"
		mkdir -p $log_dir
	fi
	echo "$dump_dir/prog_${prog}.metric_${metric_name}.noise_${noise_type}${noise_scale}.opt$opt.out"
	sbatch -A cis230379 --nodes=1 --ntasks=1 --cpus-per-task=1 <<EOT
#!/bin/bash -l

#SBATCH --job-name="VSR_MCTS${type}${nv}${nt}"
#SBATCH --output=$log_dir/${eq_name}.metric_${metric_name}.noise_${noise_type}_${noise_scale}.opt${opt}.cv_mcts.out
#SBATCH --constraint=A
#SBATCH --time=12:00:00
#SBATCH --mem=4GB

hostname
$py310 $thispath/main.py --equation_name $data_path/$eq_name --optimizer $opt --cv_mcts \
				--track_memory \
				--memray_output_bin $scratch_dir/prog_${prog}.metric_${metric_name}.noise_${noise_type}${noise_scale}.opt$opt.cv_mcts.bin \
        		--metric_name $metric_name --noise_type $noise_type --noise_scale $noise_scale > $dump_dir/prog_${prog}.metric_${metric_name}.noise_${noise_type}${noise_scale}.opt$opt.cv_mcts.out
memray stats $scratch_dir/prog_${prog}.metric_${metric_name}.noise_${noise_type}${noise_scale}.opt$opt.cv_mcts.bin --json
EOT

sbatch -A cis230379 --nodes=1 --ntasks=1 --cpus-per-task=1 <<EOT
#!/bin/bash -l

#SBATCH --job-name="MCTS${type}${nv}${nt}"
#SBATCH --output=$log_dir/${eq_name}.metric_${metric_name}.noise_${noise_type}_${noise_scale}.opt${opt}.mcts.out
#SBATCH --constraint=A
#SBATCH --time=12:00:00
#SBATCH --mem=4GB

hostname
$py310 $thispath/main.py --equation_name $data_path/$eq_name --optimizer $opt  \
				--track_memory \
				--memray_output_bin $scratch_dir/prog_${prog}.metric_${metric_name}.noise_${noise_type}${noise_scale}.opt$opt.mcts.bin \
        		--metric_name $metric_name --noise_type $noise_type --noise_scale $noise_scale > $dump_dir/prog_${prog}.metric_${metric_name}.noise_${noise_type}${noise_scale}.opt$opt.mcts.out
memray stats $scratch_dir/prog_${prog}.metric_${metric_name}.noise_${noise_type}${noise_scale}.opt$opt.mcts.bin --json
EOT
done
