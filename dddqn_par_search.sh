!/bin/bash
export HOME=/storage/ostrava1/home/stanksil
if [[ $HOSTNAME == zubat* ]]; then
        export THEANO_FLAGS=base_compiledir=/storage/brno2/home/stanksil/theanoc
fi
cd $HOME

. init_script.sh
cd DQN-for-autonomous-vehicles

python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1234 --hidden_size [20,20] --memory_steps 0 --exploration_strategy semi_uniform --save_path models/1/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1234 --hidden_size [20,20] --memory_steps 0 --exploration_strategy boltzmann --save_path models/2/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1234 --hidden_size [20,20] --memory_steps 0 --exploration_strategy e_greedy --save_path models/3/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1234 --hidden_size [20,20] --memory_steps 1 --exploration_strategy semi_uniform --save_path models/4/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1234 --hidden_size [20,20] --memory_steps 1 --exploration_strategy boltzmann --save_path models/5/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1234 --hidden_size [20,20] --memory_steps 1 --exploration_strategy e_greedy --save_path models/6/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1234 --hidden_size [20,20] --memory_steps 3 --exploration_strategy semi_uniform --save_path models/7/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1234 --hidden_size [20,20] --memory_steps 3 --exploration_strategy boltzmann --save_path models/8/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1234 --hidden_size [20,20] --memory_steps 3 --exploration_strategy e_greedy --save_path models/9/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1234 --hidden_size [20,20] --memory_steps 5 --exploration_strategy semi_uniform --save_path models/10/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1234 --hidden_size [20,20] --memory_steps 5 --exploration_strategy boltzmann --save_path models/11/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1234 --hidden_size [20,20] --memory_steps 5 --exploration_strategy e_greedy --save_path models/12/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1234 --hidden_size [50,50] --memory_steps 0 --exploration_strategy semi_uniform --save_path models/13/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1234 --hidden_size [50,50] --memory_steps 0 --exploration_strategy boltzmann --save_path models/14/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1234 --hidden_size [50,50] --memory_steps 0 --exploration_strategy e_greedy --save_path models/15/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1234 --hidden_size [50,50] --memory_steps 1 --exploration_strategy semi_uniform --save_path models/16/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1234 --hidden_size [50,50] --memory_steps 1 --exploration_strategy boltzmann --save_path models/17/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1234 --hidden_size [50,50] --memory_steps 1 --exploration_strategy e_greedy --save_path models/18/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1234 --hidden_size [50,50] --memory_steps 3 --exploration_strategy semi_uniform --save_path models/19/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1234 --hidden_size [50,50] --memory_steps 3 --exploration_strategy boltzmann --save_path models/20/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1234 --hidden_size [50,50] --memory_steps 3 --exploration_strategy e_greedy --save_path models/21/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1234 --hidden_size [50,50] --memory_steps 5 --exploration_strategy semi_uniform --save_path models/22/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1234 --hidden_size [50,50] --memory_steps 5 --exploration_strategy boltzmann --save_path models/23/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1234 --hidden_size [50,50] --memory_steps 5 --exploration_strategy e_greedy --save_path models/24/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1337 --hidden_size [20,20] --memory_steps 0 --exploration_strategy semi_uniform --save_path models/25/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1337 --hidden_size [20,20] --memory_steps 0 --exploration_strategy boltzmann --save_path models/26/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1337 --hidden_size [20,20] --memory_steps 0 --exploration_strategy e_greedy --save_path models/27/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1337 --hidden_size [20,20] --memory_steps 1 --exploration_strategy semi_uniform --save_path models/28/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1337 --hidden_size [20,20] --memory_steps 1 --exploration_strategy boltzmann --save_path models/29/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1337 --hidden_size [20,20] --memory_steps 1 --exploration_strategy e_greedy --save_path models/30/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1337 --hidden_size [20,20] --memory_steps 3 --exploration_strategy semi_uniform --save_path models/31/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1337 --hidden_size [20,20] --memory_steps 3 --exploration_strategy boltzmann --save_path models/32/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1337 --hidden_size [20,20] --memory_steps 3 --exploration_strategy e_greedy --save_path models/33/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1337 --hidden_size [20,20] --memory_steps 5 --exploration_strategy semi_uniform --save_path models/34/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1337 --hidden_size [20,20] --memory_steps 5 --exploration_strategy boltzmann --save_path models/35/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1337 --hidden_size [20,20] --memory_steps 5 --exploration_strategy e_greedy --save_path models/36/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1337 --hidden_size [50,50] --memory_steps 0 --exploration_strategy semi_uniform --save_path models/37/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1337 --hidden_size [50,50] --memory_steps 0 --exploration_strategy boltzmann --save_path models/38/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1337 --hidden_size [50,50] --memory_steps 0 --exploration_strategy e_greedy --save_path models/39/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1337 --hidden_size [50,50] --memory_steps 1 --exploration_strategy semi_uniform --save_path models/40/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1337 --hidden_size [50,50] --memory_steps 1 --exploration_strategy boltzmann --save_path models/41/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1337 --hidden_size [50,50] --memory_steps 1 --exploration_strategy e_greedy --save_path models/42/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1337 --hidden_size [50,50] --memory_steps 3 --exploration_strategy semi_uniform --save_path models/43/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1337 --hidden_size [50,50] --memory_steps 3 --exploration_strategy boltzmann --save_path models/44/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1337 --hidden_size [50,50] --memory_steps 3 --exploration_strategy e_greedy --save_path models/45/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1337 --hidden_size [50,50] --memory_steps 5 --exploration_strategy semi_uniform --save_path models/46/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1337 --hidden_size [50,50] --memory_steps 5 --exploration_strategy boltzmann --save_path models/47/&
python duel.py ACar-v0 --episodes 2011 --advantage max   --seed 1337 --hidden_size [50,50] --memory_steps 5 --exploration_strategy e_greedy --save_path models/48/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1234 --hidden_size [20,20] --memory_steps 0 --exploration_strategy semi_uniform --save_path models/49/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1234 --hidden_size [20,20] --memory_steps 0 --exploration_strategy boltzmann --save_path models/50/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1234 --hidden_size [20,20] --memory_steps 0 --exploration_strategy e_greedy --save_path models/51/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1234 --hidden_size [20,20] --memory_steps 1 --exploration_strategy semi_uniform --save_path models/52/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1234 --hidden_size [20,20] --memory_steps 1 --exploration_strategy boltzmann --save_path models/53/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1234 --hidden_size [20,20] --memory_steps 1 --exploration_strategy e_greedy --save_path models/54/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1234 --hidden_size [20,20] --memory_steps 3 --exploration_strategy semi_uniform --save_path models/55/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1234 --hidden_size [20,20] --memory_steps 3 --exploration_strategy boltzmann --save_path models/56/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1234 --hidden_size [20,20] --memory_steps 3 --exploration_strategy e_greedy --save_path models/57/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1234 --hidden_size [20,20] --memory_steps 5 --exploration_strategy semi_uniform --save_path models/58/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1234 --hidden_size [20,20] --memory_steps 5 --exploration_strategy boltzmann --save_path models/59/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1234 --hidden_size [20,20] --memory_steps 5 --exploration_strategy e_greedy --save_path models/60/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1234 --hidden_size [50,50] --memory_steps 0 --exploration_strategy semi_uniform --save_path models/61/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1234 --hidden_size [50,50] --memory_steps 0 --exploration_strategy boltzmann --save_path models/62/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1234 --hidden_size [50,50] --memory_steps 0 --exploration_strategy e_greedy --save_path models/63/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1234 --hidden_size [50,50] --memory_steps 1 --exploration_strategy semi_uniform --save_path models/64/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1234 --hidden_size [50,50] --memory_steps 1 --exploration_strategy boltzmann --save_path models/65/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1234 --hidden_size [50,50] --memory_steps 1 --exploration_strategy e_greedy --save_path models/66/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1234 --hidden_size [50,50] --memory_steps 3 --exploration_strategy semi_uniform --save_path models/67/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1234 --hidden_size [50,50] --memory_steps 3 --exploration_strategy boltzmann --save_path models/68/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1234 --hidden_size [50,50] --memory_steps 3 --exploration_strategy e_greedy --save_path models/69/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1234 --hidden_size [50,50] --memory_steps 5 --exploration_strategy semi_uniform --save_path models/70/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1234 --hidden_size [50,50] --memory_steps 5 --exploration_strategy boltzmann --save_path models/71/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1234 --hidden_size [50,50] --memory_steps 5 --exploration_strategy e_greedy --save_path models/72/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1337 --hidden_size [20,20] --memory_steps 0 --exploration_strategy semi_uniform --save_path models/73/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1337 --hidden_size [20,20] --memory_steps 0 --exploration_strategy boltzmann --save_path models/74/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1337 --hidden_size [20,20] --memory_steps 0 --exploration_strategy e_greedy --save_path models/75/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1337 --hidden_size [20,20] --memory_steps 1 --exploration_strategy semi_uniform --save_path models/76/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1337 --hidden_size [20,20] --memory_steps 1 --exploration_strategy boltzmann --save_path models/77/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1337 --hidden_size [20,20] --memory_steps 1 --exploration_strategy e_greedy --save_path models/78/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1337 --hidden_size [20,20] --memory_steps 3 --exploration_strategy semi_uniform --save_path models/79/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1337 --hidden_size [20,20] --memory_steps 3 --exploration_strategy boltzmann --save_path models/80/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1337 --hidden_size [20,20] --memory_steps 3 --exploration_strategy e_greedy --save_path models/81/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1337 --hidden_size [20,20] --memory_steps 5 --exploration_strategy semi_uniform --save_path models/82/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1337 --hidden_size [20,20] --memory_steps 5 --exploration_strategy boltzmann --save_path models/83/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1337 --hidden_size [20,20] --memory_steps 5 --exploration_strategy e_greedy --save_path models/84/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1337 --hidden_size [50,50] --memory_steps 0 --exploration_strategy semi_uniform --save_path models/85/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1337 --hidden_size [50,50] --memory_steps 0 --exploration_strategy boltzmann --save_path models/86/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1337 --hidden_size [50,50] --memory_steps 0 --exploration_strategy e_greedy --save_path models/87/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1337 --hidden_size [50,50] --memory_steps 1 --exploration_strategy semi_uniform --save_path models/88/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1337 --hidden_size [50,50] --memory_steps 1 --exploration_strategy boltzmann --save_path models/89/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1337 --hidden_size [50,50] --memory_steps 1 --exploration_strategy e_greedy --save_path models/90/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1337 --hidden_size [50,50] --memory_steps 3 --exploration_strategy semi_uniform --save_path models/91/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1337 --hidden_size [50,50] --memory_steps 3 --exploration_strategy boltzmann --save_path models/92/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1337 --hidden_size [50,50] --memory_steps 3 --exploration_strategy e_greedy --save_path models/93/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1337 --hidden_size [50,50] --memory_steps 5 --exploration_strategy semi_uniform --save_path models/94/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1337 --hidden_size [50,50] --memory_steps 5 --exploration_strategy boltzmann --save_path models/95/&
python duel.py ACar-v0 --episodes 2011 --advantage max --prioritize  --seed 1337 --hidden_size [50,50] --memory_steps 5 --exploration_strategy e_greedy --save_path models/96/&
