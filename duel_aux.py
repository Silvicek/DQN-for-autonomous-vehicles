
class ReplayHolder:
    """Container used in replay memory"""
    def __init__(self, s_t, a_t):
        self.s_t = s_t
        self.a_t = a_t
        self.r_t = 0.
        self.s_tp1 = s_t
        self.last = False
        self.delta = 1.

    def complete(self, r_t, s_tp1, last):
        self.last = last
        self.r_t = r_t
        self.s_tp1 = s_tp1


def print_results(reward, args):
    res = 'results.csv' if args.result_id is None else args.result_id+'.csv'
    path = args.save_path+'/'+res
    args.reward = reward
    d = vars(args)
    import csv

    with open(path, 'wb') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, d.keys())
        w.writeheader()
        w.writerow(d)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    YELLOW = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.YELLOW = ''
        self.FAIL = ''
        self.ENDC = ''
