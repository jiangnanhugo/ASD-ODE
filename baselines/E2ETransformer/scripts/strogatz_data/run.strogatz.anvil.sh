#!/usr/bin/zsh

basepath=/home/$USER/data/act_ode
py3=/home/$USER/workspace/miniconda3/envs/py310/bin/python3

type=Strogatz
method=e2etransformer
noise_type=normal
noise_scale=0.0
metric_name=inv_mse
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

    echo "output dir: $dump_dir/${eq_name}.noise_${noise_type}${noise_scale}.$method.out"
    sbatch -A cis230379 --nodes=1 --ntasks=1 --cpus-per-task=1 <<EOT
#!/bin/bash -l

#SBATCH --job-name="E2E-${eq_name}"
#SBATCH --output=$log_dir/${eq_name}.metric_${metric_name}.noise_${noise_type}_${noise_scale}.$method.out
#SBATCH --constraint=A
#SBATCH --time=24:00:00


hostname
$py3 $basepath/baselines/E2ETransformer/main.py --equation_name $eq_name --pretrained_model_filepath $basepath/baselines/E2ETransformer/model.pt --mode cpu \
		--metric_name $metric_name --num_init_conds $num_init_conds --noise_type $noise_type --noise_scale $noise_scale >$dump_dir/${eq_name}.noise_${noise_type}${noise_scale}.$method.out
EOT
done
