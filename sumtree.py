"""SumTree structure necessary for proportional prioritization"""
import numpy as np
from itertools import cycle


class SumTree:

    class AbstractNode:
        def __init__(self, value=None, ix=0, sum=0):
            if value is not None:
                sum = value.delta
            self.pointer = value
            self.ix = ix
            self.sum = sum

    def __init__(self):
        self.tree = []

    def add_node(self, value):
        """Replace last node in tree with new AbstractNode.
        It's children are the old and the new values"""
        if len(self.tree) == 0:
            self.tree.append(SumTree.AbstractNode(value=value))
            return self.tree[0]
        self.tree.append(SumTree.AbstractNode(value=value))
        ix = len(self.tree)-1
        left = self.tree[ix]
        left.ix = ix

        right = self.tree[(ix-1)/2]
        self.tree.append(right)
        right.ix = len(self.tree)-1

        middle = SumTree.AbstractNode(sum=left.sum+right.sum)
        self.tree[(ix-1)/2] = middle
        middle.ix = (ix-1)/2
        self.update(ix)
        return left

    def update(self, ix):
        """Step up in the tree, updating subsums along the way"""
        self.tree[ix].sum = self.tree[ix].pointer.delta
        while True:
            ix = (ix-1)/2
            self.tree[ix].sum = self.tree[2*ix+1].sum + self.tree[2*ix+2].sum
            if ix == 0:
                break

    def sample(self, target):
        """Trace a value in the tree"""
        ix = 0
        while True:
            left = 2*ix+1
            right = 2*ix+2
            if left >= len(self.tree):
                return ix
            if target <= self.tree[left].sum:
                ix = left
            else:
                ix = right
                target -= self.tree[left].sum

    def sample_random(self):
        ix = 0
        while True:
            left = 2*ix+1
            right = 2*ix+2
            if left >= len(self.tree):
                return ix
            if np.random.randint(2) == 0:
                ix = left
            else:
                ix = right

    def last_ixs(self, return_array=False):
        """Return indices of all non-abstract nodes"""
        n = len(self.tree)
        n_min = 2**int(np.log2(n))-1
        n_max = 2**int(np.log2(n)+1)-1
        r_t = n_max - n_min
        r_last = r_t - (n_max - n)
        r_pre = r_t/2-r_last/2
        ixs = r_last + r_pre
        if return_array:
            return range(n-ixs, n)
        return cycle(range(n-ixs, n))

    def print_(self):
        depth = int(np.log2(len(self.tree)))+1
        ix = 0
        nums = 1
        for j in range(depth):
            for i in range(nums):
                mul = 1 if i==0 else 2
                print '-'*int(mul*np.power(2,(depth-j))),self.tree[ix].sum,'(',self.tree[ix].ix, ix,')',
                ix += 1
                if ix == len(self.tree):
                    print
                    return
            print
            nums *= 2


if __name__ == '__main__':
    class X:
        def __init__(self):
            self.delta = np.random.randint(10)

    tree = SumTree()
    xs = []
    nodes = []
    np.random.seed(1338)
    for _ in range(3):
        x = X()
        xs.append(x)
        nodes += [tree.add_node(x)]
        # tree.print_()
        # print '=========='

    tree.print_()
    # print [int(tree.tree[i].pointer is None) for i in range(len(tree.tree))]
    #
    # for i in range(tree.tree[0].sum):
    #     print i, tree.sample_random()

    print tree.last_ixs()