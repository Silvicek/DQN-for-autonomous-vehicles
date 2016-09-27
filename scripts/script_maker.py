"""
A brute-force approach to parameter search. See evolution.py for a cooler one.
"""
script_preface = """!/bin/bash
export HOME=/storage/ostrava1/home/stanksil
if [[ $HOSTNAME == zubat* ]]; then
        export THEANO_FLAGS=base_compiledir=/storage/brno2/home/stanksil/theanoc
fi
cd $HOME

. init_script.sh
cd DQN-for-autonomous-vehicles

"""


def brute_force():
    hidden_sizes = ('--hidden_size', ['[20,20]', '[50,50]'])
    seeds = ('--seed', [1234, 1337])
    memory_steps = ('--memory_steps', [0, 1, 3, 5])
    strategies = ('--exploration_strategy', ['semi_uniform', 'boltzmann', 'e_greedy'])
    # advantage = ('--advantage', ['naive', 'avg', 'max'])

    other = ['', '--prioritize']

    args = (seeds, hidden_sizes, memory_steps, strategies)

    i = 0
    runs = []
    cpu_usage = 4

    def go_through(xxx, params):
        global i, runs
        arg, vals = xxx[0]
        for val in vals:
            if len(xxx) > 1:
                go_through(xxx[1:], params + ' ' + arg + ' ' + str(val))
            else:
                i += 1
                string = params + ' ' + arg + ' ' + str(val) + ' ' + '--save_path models/'+str(i) + '/'
                if i % cpu_usage == 0:
                    string += ' \n'
                else:
                    string += ' &\n'
                runs.append(string)

    basic_args = 'python duel.py ACar-v0 --episodes 2011 --advantage max'
    for arg in other:
        go_through(args, basic_args + ' ' + arg + ' ')

    j = 0
    while True:
        if len(runs[j*20:]) > 20:
            runz = ''.join(runs[j*20:j*20+20])
        else:
            runz = ''.join(runs[j*20:])
        script = open('dddqn_par_search_' + str(j)+'.sh', 'wr')
        script.write(script_preface + runz)
        script.close()
        j += 1

        if j*20 > len(runs):
            break


import os

def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]


def check_results(path='../models'):
    import csv
    dirs = [x[0] for x in walklevel(path)]
    print dirs

    results = []

    for dir in dirs[1:]:
        try:
            with open(dir + '/results.csv') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                i = 0
                for row in reader:
                    if i == 1:
                        results.append((int(dir.split('/')[-1]), row[-5]))
                    i += 1
        except IOError:
            results.append((int(dir.split('/')[-1]), 'MISSING'))

    results.sort()
    for x in results:
        print x[0], x[1]


check_results('models')