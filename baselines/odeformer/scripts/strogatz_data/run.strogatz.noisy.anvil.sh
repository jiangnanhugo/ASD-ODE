#!/usr/bin/zsh

basepath=/home/$USER/data/act_ode
py3=/home/$USER/workspace/miniconda3/envs/py310/bin/python3

type=Strogatz
method=odeformer
noise_type=normal
noise_scale=0.01
metric_name=neg_mse
num_init_conds=5
nvars=$1
total_progs=$2
pretrain_basepath=$basepath/baselines/odeformer/
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

#SBATCH --job-name="ODEF-${eq_name}"
#SBATCH --output=$log_dir/${eq_name}.metric_${metric_name}.noise_${noise_type}_${noise_scale}.$method.out
#SBATCH --constraint=A
#SBATCH --time=24:00:00
#SBATCH --mem=8GB

hostname
$py3 $basepath/baselines/odeformer/baseline_odeformer.py --pretrain_basepath $pretrain_basepath --equation_name $eq_name \
                --metric_name $metric_name --num_init_conds $num_init_conds --noise_type $noise_type --noise_scale $noise_scale  >$dump_dir/${eq_name}.noise_${noise_type}${noise_scale}.$method.out
EOT
done
