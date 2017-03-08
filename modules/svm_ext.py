#!/usr/bin/env python

import math
import os
from subprocess import Popen, PIPE, STDOUT
from random import seed
from bindings.libsvm.svmutil import svm_predict as svm_pred
from bindings.libsvm.svmutil import svm_load_model
from modules.util import kernel_rbf
from operator import itemgetter

odir_files = ''


# sets the output directory path
def set_odir(s):
    global odir_files
    odir_files = s

# defines the executable pathnames
svmscale_exe = './lib/libsvm-ext/bin/svm-scale'
svmtrain_exe = './lib/libsvm-ext/bin/svm-train'
svmpredict_exe = './lib/libsvm-ext/bin/svm-predict'
grid_py = 'tools/grid.py'


# checks if the executable pathnames are correct
def check_exe_pathnames():
    assert os.path.exists(svmscale_exe),\
        'svm-scale executable not found'
    assert os.path.exists(svmtrain_exe),\
        'svm-train executable not found'
    assert os.path.exists(svmpredict_exe),\
        'svm-predict executable not found'
    assert os.path.exists(grid_py),\
        'grid.py not found'

# checks the exe pathnames
check_exe_pathnames()


# scale train/test data using range
def scale_data(ds_path, out_path, range_path, load_range):
    if not (os.path.split(range_path)[1] in odir_files and
            os.path.split(out_path)[1] in odir_files):
        p = 's' if not load_range else 'r'
        cmd = '{0} -l -1 -u 1 -{1} "{2}" "{3}" > "{4}"'\
              .format(svmscale_exe, p, range_path, ds_path, out_path)
        out = Popen(cmd, shell=True, stdout=PIPE, stderr=STDOUT)\
            .communicate()[0]
        if out is not None and out != "":
            f = open(out_path + '.log', 'w')
            f.write(out)
            f.close()


# selects a subset based with the given lines
def select_subset(ds_x, ds_y, sub_lines):
    ds1_x, ds1_y, ds2_x, ds2_y = [], [], [], []
    for line in xrange(len(ds_y)):
        if line in sub_lines:
            ds2_x += [ds_x[line]]
            ds2_y += [ds_y[line]]
        else:
            ds1_x += [ds_x[line]]
            ds1_y += [ds_y[line]]
    return ds1_x, ds1_y, ds2_x, ds2_y


# multiprocess grid search for csvm
def grid_search(ds_train_path, out_path, fold, c_begin, c_end, c_step, g_begin,
                g_end, g_step, params, ssh_workers, nr_local_worker):
    cmd = '{0} -log2c {1},{2},{3} -log2g {4},{5},{6} -v {7}'
    cmd += ' -svmtrain "{8}" -out "{9}"'
    if ssh_workers:
        cmd += ' -png "{10}" -nodes "{11}" -workers {12} {13} "{14}"'
        cmd = cmd.format(
            grid_py, c_begin, c_end, c_step, g_begin, g_end, g_step, fold,
            svmtrain_exe, out_path + '.out', out_path + '.png',
            ','.join(ssh_workers), nr_local_worker, params, ds_train_path
            + '.scale')
    else:
        cmd += ' -png "{10}" -workers {11} {12} "{13}"'
        cmd = cmd.format(
            grid_py, c_begin, c_end, c_step, g_begin, g_end, g_step, fold,
            svmtrain_exe, out_path + '.out', out_path + '.png',
            nr_local_worker, params, ds_train_path + '.scale')

    Popen(cmd, shell=True, stdout=PIPE).communicate()


# returns svm opt parameters
def svm_train(ds_train_path, ds_train_scale_path, svm_fold, svm_type,
              svm_kernel, svm_cache, svm_probability, svm_weight_pos,
              svm_weight_neg, ssh_workers, workers):
    # grid search params
    params = '-s ' + str(svm_type)
    params += ' -t ' + str(svm_kernel)
    params += ' -m ' + str(svm_cache)
    params += ' -b ' + str(svm_probability)
    if svm_weight_pos != 1:
        params += ' -w1 ' + str(svm_weight_pos)
    if svm_weight_neg != 1:
        params += ' -w-1 ' + str(svm_weight_neg)

    out_path = ds_train_path
    if svm_weight_pos != 1:
        out_path += '.wpos_' + str(svm_weight_pos)
    if svm_weight_neg != 1:
        out_path += '.wneg_' + str(svm_weight_neg)

    opt_params = get_svm_opt_params(
        ds_train_path, out_path, fold=svm_fold, c_begin=-5,
        c_end=15, c_step=2, g_begin=3, g_end=-15, g_step=-2,
        params=params, ssh_workers=ssh_workers, workers=workers)

    # generates and loads the optmized trained model in memory
    model = get_model(
        ds_train_scale_path, out_path + '.model',
        opt_params['cost'], opt_params['gamma'])
    # norm of the vector w
    norm_w = model.get_norm_w()
    # size of svm margin
    margin_size = 1 / norm_w

    return model, opt_params['cost'], opt_params['gamma'],\
        opt_params['rate'], norm_w, margin_size


# recover the best cost and gamma
def get_svm_opt_params(ds_train_path, out_path, fold, c_begin, c_end, c_step,
                       g_begin, g_end, g_step, params, ssh_workers, workers):

    exists_out = os.path.split(out_path + '.out')[1] in odir_files
    exists_model = os.path.split(out_path + '.model')[1] in odir_files

    # grid search
    if not exists_out or not exists_model:
        if exists_out:
            os.remove(out_path + '.out')
        if exists_model:
            os.remove(out_path + '.model')
        grid_search(
            ds_train_path, out_path, fold, c_begin, c_end, c_step, g_begin,
            g_end, g_step, params, ssh_workers, workers)

    # getting best cost and gamma
    f = open(out_path + '.out', 'r')
    best_rate = 0
    c, g, rate = (0, 0, 0)
    line = ''
    while True:
        line = f.readline()
        if not line:
            break
        if map(float, line.split())[2] > best_rate:
            c, g, rate = map(float, line.split())
            best_rate = rate

    return {'rate': rate, 'cost': 2.0 ** c, 'gamma': 2.0 ** g}


# generate and return the trained model
def get_model(ds_train_path, out_path, cost, gamma):
    if not os.path.split(out_path)[1] in odir_files:
        cmd = '{0} -c {1} -g {2} "{3}" "{4}"'.format(
            svmtrain_exe, cost, gamma, ds_train_path, out_path)
        Popen(cmd, shell=True, stdout=PIPE).communicate()
    return svm_load_model(out_path)


# gets decision values for the input dataset
def svm_predict(ds_x, ds_y, model):
    # get decision values (positive = label+)
    seed(0)  # reset random seed
    labels = model.get_labels()
    pl, evals, dv, rt = svm_pred(ds_y, ds_x, model, '-q')
    dv = [labels[0] * val[0] for val in dv]

    return pl, dv, rt


# returns the best hyperplane bias for some fp-threshold, with np-score
def get_opt_bias(roc_points, roc_bias, pos, neg, fpt, svm_rates, svm_bias,
                 norm_w):
    opt_bias = 0
    opt_bias_pm = 2 ** 20
    # assigns the original svm rate results as the initial best bias
    if svm_rates[1] <= fpt:
        opt_bias = svm_bias
        opt_bias_pm = get_np_score(fpt, svm_rates[1], svm_rates[3])

    # searchs for better bias values
    for i in xrange(len(roc_points)):
        tpr = roc_points[i][1] / float(pos)
        fpr = roc_points[i][0] / float(neg)
        fnr = 1 - tpr
        pm = get_np_score(fpt, fpr, fnr)
        is_better = pm < opt_bias_pm
        if is_better and roc_bias[i][0] > 0:
            opt_bias = svm_bias + (roc_bias[i][0] / norm_w)
            opt_bias_pm = pm
    return opt_bias


# returns predicted labels with shift-bias
def get_pl_shift_bias(dvs, shift_bias, svm_bias, norm_w):
    # returns 1 if instance is above the shifted bias, otherwise returns -1
    return [1 if svm_bias + (dv / float(norm_w)) >= shift_bias
            else -1 for dv in dvs]


# returns decision values with shift-bias
def get_dv_shift_bias(dvs, shift_bias, svm_bias, norm_w):
    bias_diff_denorm = norm_w * (svm_bias - shift_bias)
    return [dv + bias_diff_denorm for dv in dvs]


# computes tp, fp, tn, fn rates
def get_measures(plabels, labels):
    # tp, fp, tn and fn rates
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in xrange(len(labels)):
        if int(plabels[i]) == 1:
            if int(labels[i]) == 1:
                tp += 1
            else:
                fp += 1
        else:
            if int(labels[i]) == 1:
                fn += 1
            else:
                tn += 1

    return tp, fp, tn, fn


# compute single performance measure
def get_performance_measure(tp, fp, tn, fn):
    return get_f_measure(tp, fp, fn)


# returns the f1-score
def get_f_measure(tp, fp, fn):
    precision = 0 if (tp + fp) == 0 else tp / float(tp + fp)
    recall = 0 if (tp + fn) == 0 else tp / float(tp + fn)
    f_measure = 0 if precision + recall == 0\
        else 2 * (precision * recall) / float(precision + recall)
    return f_measure


# returns the mcc score
def get_mcc(tp, fp, tn, fn):
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    den = 1 if den == 0 else den
    mcc = (tp * tn - fp * fn) / float(math.sqrt(den))
    return mcc


# returns the np-score
def get_np_score(fpt, fpr, fnr):
    return (1 / float(fpt)) * max(fpr - fpt, 0) + fnr


# compute t-auc
def get_t_auc(dv, y, fpt):
    # count of postive and negative labels
    db = []
    pos, neg = 0, 0
    for i in range(len(y)):
        if y[i] > 0:
            pos += 1
        else:
            neg += 1
        db.append([dv[i], y[i]])

    # sorting by decision value
    db = sorted(db, key=itemgetter(0), reverse=True)

    # calculate ROC
    x_fpr, y_tpr = [], []
    xy_arr = []
    xy_arr.append([0., 0.])
    tp, fp = 0., 0.         # assure float division
    for i in range(len(db)):
        if db[i][1] > 0:    # positive
            tp += 1
        else:
            fp += 1
        x_fpr.append(fp / neg)
        y_tpr.append(tp / pos)
        xy_arr.append([fp / neg, tp / pos])
    xy_arr.append([1., 1.])

    # area under curve
    tauc = 0.0
    prev_x = 0
    for x, y in xy_arr:
        if x > fpt:
            break
        if x != prev_x:
            tauc += (x - prev_x) * y
            prev_x = x

    return tauc


# calculates the distance between two points
def get_p2p_distance(p1, p2, g):
    d2 = 2.0 - (2.0 * kernel_rbf(p1, p2, g))
    return d2


# get the point-to-point distances from a specific file
def get_distances(id_orig, cur, dist, dist_type, disallowed_rows=[]):
    sql = """SELECT id_dest FROM neighbors_%s
             WHERE id_orig = ? AND max_dist = ?
             ORDER BY pos ASC""" % (dist_type,)
    cur.execute(sql, (id_orig, dist,))
    res = cur.fetchall()
    return [i[0] for i in res if i[0] not in disallowed_rows]
