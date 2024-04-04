#!/usr/bin/zsh

basepath=/depot/yexiang/apps/jiang631/data/scibench
py310=/home/jiang631/workspace/miniconda3/envs/py310/bin/python3
type=$1
thispath=$basepath/mcts_and_vsr_mcts
data_path=$basepath/data/unencrypted/equations_feynman
opt=L-BFGS-B
num_episodes=5000
noise_type=normal
noise_scale=0.0
metric_name=neg_nmse


if [[ $type -eq 2 ]]
then
echo EQUILATERAL
all_files=(FeynmanICh12Eq1.in FeynmanICh6Eq20.in FeynmanICh10Eq7.in FeynmanICh12Eq4.in FeynmanICh14Eq3.in FeynmanICh12Eq5.in FeynmanICh14Eq4.in FeynmanICh15Eq10.in FeynmanICh16Eq6.in FeynmanICh25Eq13.in FeynmanICh26Eq2.in FeynmanICh32Eq5.in FeynmanICh34Eq10.in FeynmanICh34Eq14.in FeynmanICh38Eq12.in FeynmanICh39Eq10.in FeynmanICh41Eq16.in FeynmanICh43Eq31.in FeynmanICh48Eq2.in FeynmanIICh3Eq24.in FeynmanIICh4Eq23.in FeynmanIICh8Eq7.in FeynmanIICh10Eq9.in FeynmanIICh11Eq28.in FeynmanIICh13Eq17.in FeynmanIICh13Eq23.in FeynmanIICh13Eq34.in FeynmanIICh24Eq17.in FeynmanIICh34Eq29a.in FeynmanIICh38Eq14.in FeynmanIIICh4Eq32.in FeynmanIIICh4Eq33.in FeynmanIIICh7Eq38.in FeynmanIIICh8Eq54.in FeynmanIIICh15Eq14.in FeynmanBonus8.in FeynmanBonus10.in)
elif [[ $type -eq 3 ]]
then
all_files=(FeynmanICh6Eq20b.in FeynmanICh12Eq2.in FeynmanICh15Eq3t.in FeynmanICh15Eq3x.in FeynmanBonus20.in FeynmanICh18Eq12.in FeynmanICh27Eq6.in FeynmanICh30Eq3.in FeynmanICh30Eq5.in FeynmanICh37Eq4.in FeynmanICh39Eq11.in FeynmanICh39Eq22.in FeynmanICh43Eq43.in FeynmanICh47Eq23.in FeynmanIICh6Eq11.in FeynmanIICh6Eq15b.in FeynmanIICh11Eq27.in FeynmanIICh15Eq4.in FeynmanIICh15Eq5.in FeynmanIICh21Eq32.in FeynmanIICh34Eq2a.in FeynmanIICh34Eq2.in FeynmanIICh34Eq29b.in FeynmanIICh37Eq1.in FeynmanIIICh13Eq18.in FeynmanIIICh15Eq12.in FeynmanIIICh15Eq27.in FeynmanIIICh17Eq37.in FeynmanIIICh19Eq51.in FeynmanBonus5.in FeynmanBonus7.in FeynmanBonus9.in FeynmanBonus15.in FeynmanBonus18.in)
elif [[ $type -eq 4 ]]
then
all_files=(FeynmanICh8Eq14.in FeynmanICh13Eq4.in FeynmanICh13Eq12.in FeynmanICh18Eq4.in FeynmanICh18Eq16.in FeynmanICh24Eq6.in FeynmanICh29Eq16.in FeynmanICh32Eq17.in FeynmanICh34Eq8.in FeynmanICh40Eq1.in FeynmanICh43Eq16.in FeynmanICh44Eq4.in FeynmanICh50Eq26.in FeynmanIICh11Eq20.in FeynmanIICh34Eq11.in FeynmanIICh35Eq18.in FeynmanIICh35Eq21.in FeynmanIICh38Eq3.in FeynmanIIICh10Eq19.in FeynmanIIICh14Eq14.in FeynmanIIICh21Eq20.in FeynmanBonus1.in FeynmanBonus3.in FeynmanBonus11.in FeynmanBonus19.in)
elif [[ $type -eq 5 ]]
then
all_files=(FeynmanICh12Eq11.in FeynmanIICh2Eq42.in FeynmanIICh6Eq15a.in FeynmanIICh11Eq3.in FeynmanIICh11Eq17.in FeynmanIICh36Eq38.in FeynmanIIICh9Eq52.in FeynmanBonus4.in FeynmanBonus12.in FeynmanBonus13.in FeynmanBonus14.in FeynmanBonus16.in)
elif [[ $type -eq 678 ]]
then
all_files=(FeynmanICh11Eq19.in FeynmanBonus2.in FeynmanBonus17.in FeynmanBonus6.in FeynmanICh9Eq18.in)
else
echo "Incorrect input"
fi

for eq_name in $all_files; do
	echo "submit $eq_name"
	trimed_name=${eq_name:7:-3}
	dump_dir=$basepath/result/Feynman_vars$type/$(date +%F)
	if [ ! -d "$dump_dir" ]; then
		echo "create dir: $dump_dir"
		mkdir -p $dump_dir
	fi
	log_dir=$basepath/log/$(date +%F)
	if [ ! -d "$log_dir" ]; then
		echo "create dir: $log_dir"
		mkdir -p $log_dir
	fi
	sbatch -A yexiang --nodes=1 --ntasks=1 --cpus-per-task=1 <<EOT
#!/bin/bash -l

#SBATCH --job-name="mtcs_${type}_$trimed_name"
#SBATCH --output=$log_dir/${eq_name}.metric_${metric_name}.noise_${noise_type}_${noise_scale}.mcts.out
#SBATCH --constraint=A
#SBATCH --time=12:00:00
#SBATCH --mem=4GB

hostname

$py310 $thispath/main.py --equation_name $data_path/$eq_name --optimizer $opt \
						--metric_name $metric_name --noise_type $noise_type --noise_scale $noise_scale \
						--num_episodes $num_episodes \
         				> $dump_dir/${eq_name}.metric_${metric_name}.noise_${noise_type}${noise_scale}.mcts.out

EOT

done
