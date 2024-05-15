# 1. noiseless setting
## 1.1 MCTS model
./run_mcts_bell.sh inv 3 22
./run_mcts_bell.sh inv 4 46
./run_mcts_bell.sh inv 5 55
./run_mcts_bell.sh inv 5 58
./run_mcts_bell.sh inv 6 68
./run_mcts_bell.sh inv 6 610

./run_mcts_bell.sh sincos 3 22
./run_mcts_bell.sh sincos 4 46
./run_mcts_bell.sh sincos 5 55
./run_mcts_bell.sh sincos 5 58
./run_mcts_bell.sh sincos 6 68
./run_mcts_bell.sh sincos 6 610

./run_mcts_bell.sh sincosinv 3 22
./run_mcts_bell.sh sincosinv 4 46
./run_mcts_bell.sh sincosinv 5 55
./run_mcts_bell.sh sincosinv 5 58
./run_mcts_bell.sh sincosinv 6 68
./run_mcts_bell.sh sincosinv 6 610

## 1.2 VSR-MCTS model
./run_cv_mcts_bell.sh sincosinv 6 68 10
./run_cv_mcts_bell.sh sincosinv 6 610 10

# 2 noisy settings
## 2.2 VSR-MCTS model
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
