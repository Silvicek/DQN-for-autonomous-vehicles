

hidden_sizes = ('--hidden_size', ['[20,20]', '[50,50]'])
seeds = ('--seed', [1234, 1337])
memory_steps = ('--memory_steps', [0, 1, 3, 5])
strategies = ('--exploration_strategy', ['semi_uniform', 'boltzmann', 'e_greedy'])
# advantage = ('--advantage', ['naive', 'avg', 'max'])

other = ['', '--prioritize']


args = (seeds, hidden_sizes, memory_steps, strategies)

i = 0
indices = range(len(args))

runs = ''

def go_through(xxx, params):
    global i, runs
    arg, vals = xxx[0]
    for val in vals:
        if len(xxx) > 1:
            go_through(xxx[1:], params + ' ' + arg + ' ' + str(val))
        else:
            i += 1
            runs += params + ' ' + arg + ' ' + str(val) + ' ' + '--save_path models/'+str(i) + '/' + '&\n'


basic_args = 'python duel.py ACar-v0 --episodes 2011 --advantage max'
for arg in other:
    go_through(args, basic_args + ' ' + arg + ' ')


script_preface = """!/bin/bash
export HOME=/storage/ostrava1/home/stanksil
if [[ $HOSTNAME == zubat* ]]; then
        export THEANO_FLAGS=base_compiledir=/storage/brno2/home/stanksil/theanoc
fi
cd $HOME

. init_script.sh
cd DQN-for-autonomous-vehicles

"""


script = open('dddqn_par_search.sh', 'wr')

script.write(script_preface + runs)

script.close()