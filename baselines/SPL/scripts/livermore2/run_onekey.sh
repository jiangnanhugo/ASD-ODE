# 1. noiseless setting
## 1.1 MCTS model
./run_mcts_vars_bell.sh 2
./run_mcts_vars_bell.sh 3
./run_mcts_vars_bell.sh 4
./run_mcts_vars_bell.sh 5
./run_mcts_vars_bell.sh 6
./run_mcts_vars_bell.sh 7


## 1.2 CV-MCTS model

./run_cv_mcts_vars_bell.sh 2
./run_cv_mcts_vars_bell.sh 3
./run_cv_mcts_vars_bell.sh 4
./run_cv_mcts_vars_bell.sh 5
./run_cv_mcts_vars_bell.sh 6
./run_cv_mcts_vars_bell.sh 7


# 2 noisy settings
## 2.2 CV-MCTS model
./run_cv_mcts_bell_noisy.sh inv 2 11
./run_cv_mcts_bell_noisy.sh inv 3 22
./run_cv_mcts_bell_noisy.sh inv 4 46
./run_cv_mcts_bell_noisy.sh inv 5 55
./run_cv_mcts_bell_noisy.sh inv 5 58
./run_cv_mcts_bell_noisy.sh inv 6 68
./run_cv_mcts_bell_noisy.sh inv 6 610

./run_cv_mcts_bell_noisy.sh sincos 2 11
./run_cv_mcts_bell_noisy.sh sincos 3 22
./run_cv_mcts_bell_noisy.sh sincos 4 46
./run_cv_mcts_bell_noisy.sh sincos 5 55
./run_cv_mcts_bell_noisy.sh sincos 5 58
./run_cv_mcts_bell_noisy.sh sincos 6 68
./run_cv_mcts_bell_noisy.sh sincos 6 610

./run_cv_mcts_bell_noisy.sh sincosinv 2 11
./run_cv_mcts_bell_noisy.sh sincosinv 3 22
./run_cv_mcts_bell_noisy.sh sincosinv 4 46
./run_cv_mcts_bell_noisy.sh sincosinv 5 55
./run_cv_mcts_bell_noisy.sh sincosinv 5 58
./run_cv_mcts_bell_noisy.sh sincosinv 6 68
./run_cv_mcts_bell_noisy.sh sincosinv 6 610
