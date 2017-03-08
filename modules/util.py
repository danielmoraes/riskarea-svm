from operator import itemgetter
import math
import sys
import random
import datetime
from collections import defaultdict
from bisect import bisect_left


# get standard deviation of a list
def stddev(l):
    l_avg = avg(l)
    return math.sqrt(
        (1 / (float(len(l)) - 1)) * math.fsum(((v - l_avg) ** 2) for v in l))


# get average value of a list
def avg(l):
    return math.fsum([v for v in l]) / float(len(l))


# rbf kernel
def kernel_rbf(i, j, g):
    # return Decimal(str(-g * (dot(i, i) + dot(j, j) - 2 * dot(i, j)))).exp()
    # return mpmath.exp(-g * (dot(i, i) + dot(j, j) - 2 * dot(i, j)))
    return -g * (dot(i, i) + dot(j, j) - 2 * dot(i, j))


# linear kernel
def kernel_linear(i, j):
    return Decimal(dot(i, j))


# inner product of two sparse vectors (dict)
def dot(i, j):
    dsum = 0
    i_keys = i.keys()
    j_keys = j.keys()
    i_pos = 0
    j_pos = 0
    while i_pos < len(i_keys) and j_pos < len(j_keys):
        if i_keys[i_pos] == j_keys[j_pos]:
            dsum += i[i_keys[i_pos]] * j[j_keys[j_pos]]
            i_pos += 1
            j_pos += 1
        else:
            if i_keys[i_pos] > j_keys[j_pos]:
                j_pos += 1
            else:
                i_pos += 1

    return dsum


# compute roc points
def compute_roc(deci, label):
    # count of postive and negative labels
    db = []
    for i in xrange(len(label)):
        db.append([deci[i], label[i]])
    # sorting by decision value
    db = sorted(db, key=itemgetter(0), reverse=True)

    # calculate roc points
    xy_arr = []
    tp, fp = 0., 0.         # assure float division
    for i in xrange(len(db)):
        if db[i][1] > 0:      # positive
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp, tp])

    # area under curve
    aoc = 0.
    prev_x = 0
    for x, y in xy_arr:
        if x != prev_x:
            aoc += (x - prev_x) * y
            prev_x = x

    return xy_arr, db, aoc


# convert libsvm sample (dict) to array
def dict_to_array(s_dict, s_size):
    s_array = [0] * s_size
    for k in s_dict:
        s_array[k - 1] = s_dict[k]
    return s_array


# sorts the original list and return the original indexes of the sorted list
def qsort_idx(l, l_idx=None):
    if l == []:
        return [[], []]
    else:
        if l_idx is None:
            l_idx = [val for val in xrange(len(l))]

        pivot = l[0]
        pivot_idx = l_idx[0]

        lesser, lesser_idx =\
            qsort_idx([x for x in l[1:] if x < pivot],
                      [l_idx[i] for i in xrange(1, len(l_idx))
                       if l[i] < pivot])
        greater, greater_idx =\
            qsort_idx([x for x in l[1:] if x >= pivot],
                      [l_idx[i] for i in xrange(1, len(l_idx))
                       if l[i] >= pivot])

        return lesser + [pivot] + greater, lesser_idx + [pivot_idx] +\
            greater_idx


# random selection
def random_selection(dataset, subset_size):
    l = sum(1 for line in open(dataset, 'r'))
    return sorted(random.sample(xrange(l), subset_size))


# random lines selection
def random_lines_selection(lines, subset_size):
    random.seed()
    return sorted(random.sample(lines, subset_size))


# stratified selection
def stratified_selection(dataset, subset_size):
    labels = [line.split(None, 1)[0] for line in dataset]
    label_linenums = defaultdict(list)
    for i, label in enumerate(labels):
        label_linenums[label] += [i]

    l = len(labels)
    remaining = subset_size
    ret = []

    # classes with fewer data are sampled first; otherwise
    # some rare classes may not be selected
    for label in sorted(label_linenums, key=lambda x: len(label_linenums[x])):
        linenums = label_linenums[label]
        label_size = len(linenums)
        # at least one instance per class
        s = int(min(remaining,
                    max(1, math.ceil(label_size * (float(subset_size) / l)))))
        if s == 0:
            sys.stderr.write('Error: failed to have at least one instance \
                             per class')
            sys.exit(-1)
        remaining -= s
        ret += [linenums[i] for i in random.sample(xrange(label_size), s)]
    return sorted(ret)


# stratified lines selection
def stratified_lines_selection(lines, labels, subset_size):
    random.seed()
    label_linenums = defaultdict(list)
    for i, label in enumerate(labels):
        label_linenums[label] += [lines[i]]

    l = len(labels)
    remaining = subset_size
    ret = []

    # classes with fewer data are sampled first; otherwise
    # some rare classes may not be selected
    for label in sorted(label_linenums, key=lambda x: len(label_linenums[x])):
        linenums = label_linenums[label]
        label_size = len(linenums)
        # at least one instance per class
        s = int(min(remaining,
                    max(1, math.ceil(label_size * (float(subset_size) / l)))))
        if s == 0:
            sys.stderr.write('Error: failed to have at least one instance \
                             per class')
            sys.exit(-1)
        remaining -= s
        ret += [linenums[i] for i in random.sample(xrange(label_size), s)]
    return sorted(ret)


# count pos neg values
def count_pos_neg(classes):
    pos, neg = 0., 0.
    for c in classes:
        if int(c) == 1:
            pos += 1
        else:
            neg += 1
    return pos, neg


# format time
def format_time(s):
    ftime = str(datetime.timedelta(seconds=s)).split(':')
    if len(ftime[0].split('day')) > 1:
        ftime[0] = int(ftime[0].split(' day')[0]) * 24 +\
            int(ftime[0].split(', ')[1])
    d = [float(i) for i in ftime]
    time = []
    if d[0] > 0:
        time.append('%dh' % (d[0]))
    if d[1] > 0:
        time.append('%dm' % (d[1]))
    if d[2] > 0:
        time.append('%0.3fs' % (d[2]))
    return ' '.join(time)


# binary search
def binary_search(a, x, lo=0, hi=None):    # can't use a to specify hi default
    hi = hi if hi is not None else len(a)  # hi defaults to len(a)
    pos = bisect_left(a, x, lo, hi)        # find insertion position
    return (pos if pos != hi and a[pos] == x else -1)  # don't walk off the end
