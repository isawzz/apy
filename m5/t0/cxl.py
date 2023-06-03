import omnibelt
from itertools import chain, combinations, product
import numpy as np


def hallo():
    print("hallo")


def is_string(x):
    return type(x) == "str"


def make_adict(skeys, vals):
    x = omnibelt.adict()
    keys = skeys.split()
    for i in range(len(keys)):
        x[keys[i]] = vals[i]
    return x


def powerset(iterable, lmin, lmax):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    if not lmin:
        lmin = 0
    if not lmax:
        lmax = len(s)
    return list(chain.from_iterable(combinations(s, r) for r in range(lmin, lmax + 1)))


def cartesian(*args):
    return list(product(*args))


def cartesian_fold(it):
    n = len(it[0])
    # print(n)
    sets = []
    for i in range(n):
        # print(i)
        lst = [x[i] for x in it]
        lst = list(set(lst))
        sets.append(lst)
    return sets
