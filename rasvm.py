#!/usr/bin/env python

import sys
import os, math, random, time, traceback, shutil, functools
import sqlite3 as lite
from threading import Thread
from subprocess import Popen, PIPE
from bindings.libsvm.svmutil import svm_read_problem
from modules.util import stratified_lines_selection, count_pos_neg,\
    format_time, stddev, avg, kernel_rbf, compute_roc
from modules.svm_ext import set_odir, scale_data, svm_train, get_measures,\
    get_performance_measure, svm_predict, get_dv_shift_bias, get_np_score,\
    get_opt_bias, get_pl_shift_bias
from modules.svm_knn import compute_svmknn_cross_rates,\
    get_best_svmknn_params, compute_svmknn_rates

if(sys.hexversion < 0x03000000):
    import Queue
else:
    import queue as Queue


# Main
class main():
    # --- general -----------------------------------------------------------

    # false positive thresholds to be used
    fp_thresholds = [0.1, 0.05, 0.01]

    # --- dataset -----------------------------------------------------------

    # train/test permutations
    ds_permuts = 5
    ds_reversal = True
    split_factor = 0.5

    # train and test datasets
    train_y, train_x, test_x, test_y = None, None, None, None

    # dataset positive classes
    pos_classes = [1]
    # classes used in training
    train_classes = []

    # dataset metadata
    total_train, total_test, train_lines, test_lines = None, None, None, None
    train_pos, train_neg, test_pos, test_neg = None, None, None, None

    # --- grid search -------------------------------------------------------

    # svm type (c-svc)
    svm_type = 0
    # svm kernel (radial basis kernel)
    svm_kernel = 2
    # cache size used on svm-train (mb)
    svm_cache = 300
    # probability estimates on svm-predict (false)
    svm_probability = 0
    # svm weight for pos class
    svm_weight_pos = 1
    # svm weight for neg class
    svm_weight_neg = 1

    # grid search results
    model, opt_gamma, opt_cost, opt_rate = None, None, None, None
    norm_w, bias, margin_size = None, None, None

    # --- rasvm -----------------------------------------------------------

    # rasvm margin factors
    margin_factors = [0, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25,
                      0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3, 4, 5,
                      10]
    # margin_factors = [1000]
    # rasvm k values for knn
    knn_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 17, 19, 21, 23, 25,
                  30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
                  110, 120, 130, 140, 150]

    # minimum rate to a class be positive on rasvm
    pos_min_rate = 1

    # how big k can be?
    closest_points_depth = max(knn_values)

    # --- cross validation settings -----------------------------------------

    # svm cv fold
    svm_fold = 5
    # rasvm cv fold
    svmext_fold = 5

    # --- parallel computing settings ---------------------------------------

    # ssh workers
    ssh_workers = []
    # number of local workers
    nr_local_worker = 12

    # standard contructor
    def __init__(self):
        # checks params syntax
        self.check_params()
        # sets the external processing nodes
        nodes = []
        self.set_ssh_workers(nodes)

        for pc_idx in xrange(len(self.pos_classes)):
            # checking dataset pathnames
            assert os.path.exists(sys.argv[1]), 'training file not found'

            # reads pathnames from stdin
            self.pn = self.get_pathnames(pc_idx)

            # if a testing set was not specified, run dynamic 5x2 cv
            if self.pn['ds_test_name'] is None:
                # stores the results from each fold
                partial_results = []
                self.pn['source_ds_path'] = self.pn['ds_path']
                self.pn['source_ds_train_name'] = self.pn['ds_train_name']
                for i in xrange(self.ds_permuts):
                    print '\n> INITIAL CONFIGURATION - ROUND %d/%d' %\
                          (i + 1, self.ds_permuts)

                    shutil.rmtree(self.pn['out_path_data'])
                    os.makedirs(self.pn['out_path_data'])

                    ds_out_path = (self.pn['out_path_data'] +
                                   self.pn['ds_train_name']).split('_train')[0]
                    ds_train_out_path = ds_out_path + '_train'
                    ds_test_out_path = ds_out_path + '_test'

                    # split data in the middle,
                    # which will be used as training and testing set
                    self.split_data(self.pn['source_ds_path'] + '/' +
                                    self.pn['source_ds_train_name'],
                                    ds_train_out_path, ds_test_out_path,
                                    self.split_factor)

                    self.pn['ds_path'] = self.pn['out_path_data']

                    for i in xrange(1 + 1 * self.ds_reversal):
                        # swap train and test data on the second round
                        if i == 1:
                            shutil.move(ds_train_out_path,
                                        ds_test_out_path + '.temp')
                            shutil.move(ds_test_out_path, ds_train_out_path)
                            shutil.move(ds_test_out_path + '.temp',
                                        ds_test_out_path)

                        if ds_train_out_path[-2] == '.':
                            dest_p = ds_train_out_path[:-2]
                            shutil.move(ds_train_out_path, dest_p)
                            ds_train_out_path = dest_p
                        if ds_test_out_path[-2] == '.':
                            dest_p = ds_test_out_path[:-2]
                            shutil.move(ds_test_out_path, dest_p)
                            ds_test_out_path = dest_p

                        sufix = '.' + str(i + 1)
                        shutil.move(ds_train_out_path,
                                    ds_train_out_path + sufix)
                        ds_train_out_path += sufix
                        shutil.move(ds_test_out_path,
                                    ds_test_out_path + sufix)
                        ds_test_out_path += sufix

                        self.pn['ds_train_name'] =\
                            os.path.split(ds_train_out_path)[1]
                        self.pn['ds_test_name'] =\
                            os.path.split(ds_test_out_path)[1]

                        # sets std parameters
                        self.set_std_pathnames(pc_idx)

                        # sets the data output directory
                        self.odir = os.listdir(self.pn['ds_path'])
                        set_odir(self.odir)
                        self.adjusts_the_data(pc_idx)
                        self.loads_data_in_memory()
                        partial_results.append(self.compute_rates())

                # removes the suffix indicating the round number on the file
                if ds_train_out_path[-2] == '.':
                    dest_p = ds_train_out_path[:-2]
                    shutil.move(ds_train_out_path, dest_p)
                    ds_train_out_path = dest_p
                if ds_test_out_path[-2] == '.':
                    dest_p = ds_test_out_path[:-2]
                    shutil.move(ds_test_out_path, dest_p)
                    ds_test_out_path = dest_p
                self.pn['ds_train_name'] =\
                    os.path.split(ds_train_out_path)[1]
                self.pn['ds_test_name'] =\
                    os.path.split(ds_test_out_path)[1]
                self.set_std_pathnames(pc_idx)

                results = self.compute_average_results(partial_results)
                self.export_results(results, True)
            else:
                print '\n> INITIAL CONFIGURATION'

                # sets std parameters
                self.set_std_pathnames(pc_idx)

                # clear the tmp dir
                shutil.rmtree(self.pn['out_path_data_tmp'])
                os.makedirs(self.dist_path_tmp)

                if os.path.exists('%s.sqlite' % self.dist_out_path):
                    shutil.copy('%s.sqlite' % self.dist_out_path,
                                '%s.sqlite' % self.dist_out_path_tmp)

                # sets the data output directory
                self.odir = os.listdir(self.pn['out_path_data'])
                set_odir(self.odir)

                self.adjusts_the_data(pc_idx)
                self.loads_data_in_memory()

                results = self.compute_rates()
                self.export_results(results, False)

    def split_data(self, ds_path, out_train_path, out_test_path, split_factor):
        p_wc = Popen(['wc', '-l', ds_path], stdout=PIPE)
        stdout, stderr = p_wc.communicate()

        ds_rows = int(stdout.split(' ')[0])
        sub_size = int(round(ds_rows * split_factor))

        p_subset = Popen(['python', 'tools/subset.py', ds_path, str(sub_size),
                         out_test_path, out_train_path], stdout=PIPE)
        stdout, stderr = p_subset.communicate()

    # computes the avg and std dev of the results
    def compute_average_results(self, pr):
        num_permuts = len(pr)
        strategies = pr[0].keys()

        results = {}
        for s in strategies:
            if len(pr[0][s].keys()) == 0:
                continue
            results[s] = {}
            if s == 'svm':
                # adds train_rates and test_rates avg values
                results[s]['train_rates'], results[s]['test_rates'] = [], []

                # get all the measure values
                num_measures = len(pr[0][s]['train_rates'])
                train_v = [[] for i in xrange(num_measures)]
                test_v = [[] for i in xrange(num_measures)]
                for perm in xrange(num_permuts):
                    for i in xrange(num_measures):
                        train_v[i].append(pr[perm][s]['train_rates'][i])
                        test_v[i].append(pr[perm][s]['test_rates'][i])

                # adds the avg measures to the result vec
                for i in xrange(num_measures):
                    avg_train_i = [avg(train_v[i]), stddev(train_v[i])]
                    avg_test_i = [avg(test_v[i]), stddev(test_v[i])]

                    results[s]['train_rates'].append(avg_train_i)
                    results[s]['test_rates'].append(avg_test_i)
            else:
                # adds train_rates and test_rates avg values
                results[s]['train_rates'] =\
                    [[] for i in xrange(len(self.fp_thresholds))]
                results[s]['test_rates'] =\
                    [[] for i in xrange(len(self.fp_thresholds))]

                for f in xrange(len(self.fp_thresholds)):
                    # get all the measure values
                    num_measures = len(pr[0][s]['train_rates'][0])
                    train_v = [[] for i in xrange(num_measures)]
                    test_v = [[] for i in xrange(num_measures)]
                    for perm in xrange(num_permuts):
                        for i in xrange(num_measures):
                            train_v[i].append(
                                pr[perm][s]['train_rates'][f][i])
                            test_v[i].append(
                                pr[perm][s]['test_rates'][f][i])

                    # adds the avg measures to the result vec
                    for i in xrange(num_measures):
                        avg_train_i = [avg(train_v[i]), stddev(train_v[i])]
                        avg_test_i = [avg(test_v[i]), stddev(test_v[i])]

                        results[s]['train_rates'][f].append(avg_train_i)
                        results[s]['test_rates'][f].append(avg_test_i)

                # adds opt_m and opt_k avg values
                if s in ['rasvm', 'rasvm_os', 'srasvm', 'srasvm_os',
                         'rasvm_sv', 'rasvm_os_sv', 'srasvm_sv',
                         'srasvm_os_sv']:
                    results[s]['opt_m'] = [0, 0, 0]
                    results[s]['opt_k'] = [0, 0, 0]

                    for f in xrange(len(self.fp_thresholds)):
                        # get all the tauc values
                        m_v, k_v = [], []
                        for perm in xrange(num_permuts):
                            m_v.append(pr[perm][s]['opt_m'][f])
                            k_v.append(pr[perm][s]['opt_k'][f])

                        # adds the avg tauc to the result vec
                        m = [avg(m_v), stddev(m_v)]
                        k = [avg(k_v), stddev(k_v)]
                        results[s]['opt_m'][f] = m
                        results[s]['opt_k'][f] = k

        return results

    def adjusts_the_data(self, pc_idx):
        # --- scales the data -------------------------------------------

        # starts the timer
        time_ini = self.start_timer('. scaling train data ...')
        # scales train data
        scale_data(
            self.ds_train_path, self.ds_train_scale_path,
            self.ds_train_out_path_data + '.range', False)
        # ends the timer
        self.end_timer(time_ini)

        if self.ds_test_path is not None:
            # starts the timer
            time_ini = self.start_timer('. scaling test data ...')
            # scales test data
            scale_data(
                self.ds_test_path, self.ds_test_scale_path,
                self.ds_train_out_path_data + '.range', True)
            # ends the timer
            self.end_timer(time_ini)

        # --- fixes dataset classes -------------------------------------

        # starts the timer
        time_ini = self.start_timer('. fixing dataset classes ...')

        # if the classes used in training is a subset of all classes
        # generates the train data file with only the selected classes,
        if self.train_classes:
            ds_path = self.ds_train_out_path_data + '.train-classes'
            ds_scale_path = ds_path + '.scale'
            if os.path.split(ds_scale_path)[1] not in self.odir:
                f = open(ds_scale_path, 'w')
                for line in open(self.ds_train_scale_path, 'r'):
                    line_class = line.split(None, 1)[0]
                    if int(line_class) in self.train_classes:
                        f.write(line)
                f.close()
            self.ds_train_out_path_data = ds_path
            self.ds_train_scale_path = ds_scale_path

        # fixes classes from train data
        # pos class = 1; negative classes = -1
        ds_path = self.ds_train_out_path_data + '.class-fix'
        ds_scale_path = ds_path + '.scale'
        if os.path.split(ds_scale_path)[1] not in self.odir:
            f = open(ds_scale_path, 'w')
            cont = 0
            for line in open(self.ds_train_scale_path, 'r'):
                line_class = line.split(None, 1)[0]
                if len(line.split(None, 1)) > 1:
                    line_features = line.split(None, 1)[1]
                    pc = self.pos_classes[pc_idx]
                    line_class_fix =\
                        1 if int(line_class) == int(pc) else -1
                    f.write(str(line_class_fix) + ' ' + line_features)
                cont += 1
            f.close()
        self.ds_train_out_path_data = ds_path
        self.ds_train_scale_path = ds_scale_path

        # fixes classes from test data
        # pos class = 1; negative classes = -1
        ds_path = self.ds_test_out_path_data + '.class-fix'
        ds_scale_path = ds_path + '.scale'
        if os.path.split(ds_scale_path)[1] not in self.odir:
            f = open(ds_scale_path, 'w')
            for line in open(self.ds_test_scale_path, 'r'):
                if len(line.split(None, 1)) > 1:
                    line_class = line.split(None, 1)[0]
                    line_features = line.split(None, 1)[1]
                    pc = self.pos_classes[pc_idx]
                    line_class_fix =\
                        1 if int(line_class) == int(pc) else -1
                    f.write(str(line_class_fix) + ' ' + line_features)
            f.close()
        self.ds_test_out_path_data = ds_path
        self.ds_test_scale_path = ds_scale_path

        # ends the timer
        self.end_timer(time_ini)

    def loads_data_in_memory(self):
        # starts the timer
        time_ini =\
            self.start_timer('. loading fixed dataset in memory ...')
        self.train_y, self.train_x = svm_read_problem(
            self.ds_train_scale_path)
        self.test_y, self.test_x = svm_read_problem(
            self.ds_test_scale_path)
        # ends the timer
        self.end_timer(time_ini)

    # compute rates
    def compute_rates(self):
        # gets the dataset metadata
        self.total_train = len(self.train_y)
        self.total_test = len(self.test_y)
        self.train_lines = [i for i in xrange(self.total_train)]
        self.test_lines = [i for i in xrange(self.total_test)]

        # gets the total of pos and neg values on train and test data
        self.train_pos, self.train_neg = count_pos_neg(self.train_y)
        self.test_pos, self.test_neg = count_pos_neg(self.test_y)

        # defines the subsets to be used in cross validation
        self.subsets_lines = self.stratified_subsets_lines(self.train_y,
                                                           self.svmext_fold)

        svm_results, rasvm_results, rasvm_os_results, srasvm_results,\
            srasvm_os_results, rasvm_sv_results, rasvm_os_sv_results,\
            srasvm_sv_results, srasvm_os_sv_results = [{} for i in range(9)]

        # --- GETTING OPTIMIZED PARAMETERS ----------------------------------

        print '\n> GETTING SVM OPTIMIZED PARAMETERS'

        # starts the timer
        time_ini = self.start_timer('. getting opt params ...')
        # getting optimized svm parameters
        self.model, self.opt_cost,\
            self.opt_gamma, self.opt_rate,\
            self.norm_w, self.margin_size = svm_train(
                self.ds_train_out_path_data, self.ds_train_scale_path,
                self.svm_fold, self.svm_type, self.svm_kernel,
                self.svm_cache, self.svm_probability, self.svm_weight_pos,
                self.svm_weight_neg, self.ssh_workers, self.nr_local_worker)
        # ends the timer
        self.end_timer(time_ini)

        # prints the opt. parameters returned by grid search
        print '. . c: {0}, g: {1}, cv acc: {2}'.format(
            self.opt_cost, self.opt_gamma, self.opt_rate)

        # sets the hyperplane bias returned by grid search
        self.bias = -self.model.get_rho()

        # --- ORIGINAL SVM RATES --------------------------------------------

        print '\n> COMPUTING RATE RESULTS FOR ORIGINAL SVM'
        # starts the timer
        time_ini = self.start_timer('')
        svm_results = self.compute_svm_rates(self.model, True, self.svm_type)
        # limits the training and testing rates to 10 decimal cases
        svm_results['train_rates'] = map(functools.partial(round, ndigits=10),
                                         svm_results['train_rates'])
        svm_results['test_rates'] = map(functools.partial(round, ndigits=10),
                                        svm_results['test_rates'])
        # ends the timer
        sys.stdout.write('. total time to compute svm rates')
        self.end_timer(time_ini)

        # --- RASVM RATES -------------------------------------------------

        bs_results = self.compute_bias_shift_rates(svm_results['dv_train'],
                                                   svm_results['dv_test'],
                                                   svm_results['train_rates'])

        rasvm_full_dist = 0  # rasvm, all the training points to knn

        # starts the timer
        time_ini = self.start_timer('')
        rasvm_results, rasvm_os_results, srasvm_results, srasvm_os_results =\
            self.compute_svm_knn_rates(svm_results['dv_train'],
                                       svm_results['dv_test'],
                                       svm_results['pl_train'],
                                       svm_results['pl_test'],
                                       bs_results['opt_bias'],
                                       rasvm_full_dist)

        # limits the training and testing rates to 10 decimal cases
        for i in xrange(len(self.fp_thresholds)):
            rasvm_results['train_rates'][i] = map(
                functools.partial(round, ndigits=10),
                rasvm_results['train_rates'][i])
            rasvm_results['test_rates'][i] = map(
                functools.partial(round, ndigits=10),
                rasvm_results['test_rates'][i])

            rasvm_os_results['train_rates'][i] = map(
                functools.partial(round, ndigits=10),
                rasvm_os_results['train_rates'][i])
            rasvm_os_results['test_rates'][i] = map(
                functools.partial(round, ndigits=10),
                rasvm_os_results['test_rates'][i])

            srasvm_results['train_rates'][i] = map(
                functools.partial(round, ndigits=10),
                srasvm_results['train_rates'][i])
            srasvm_results['test_rates'][i] = map(
                functools.partial(round, ndigits=10),
                srasvm_results['test_rates'][i])

            srasvm_os_results['train_rates'][i] = map(
                functools.partial(round, ndigits=10),
                srasvm_os_results['train_rates'][i])
            srasvm_os_results['test_rates'][i] = map(
                functools.partial(round, ndigits=10),
                srasvm_os_results['test_rates'][i])

        # prints the opt. parameters returned by rasvm
        print '. rasvm opt params'
        for fpt_idx in xrange(len(self.fp_thresholds)):
            print '. . fpt {0}, m: {1}, k: {2}, p: {3}'.format(
                self.fp_thresholds[fpt_idx],
                self.margin_factors[rasvm_results["opt_m"][fpt_idx]],
                self.knn_values[rasvm_results["opt_k"][fpt_idx]],
                '%.3f' % rasvm_results["test_rates"][fpt_idx][4])

        # prints the opt. parameters returned by rasvm-os
        print '. rasvm-os opt params'
        for fpt_idx in xrange(len(self.fp_thresholds)):
            print '. . fpt {0}, m: {1}, k: {2}, p: {3}'.format(
                self.fp_thresholds[fpt_idx],
                self.margin_factors[rasvm_os_results["opt_m"][fpt_idx]],
                self.knn_values[rasvm_os_results["opt_k"][fpt_idx]],
                '%.3f' % rasvm_os_results["test_rates"][fpt_idx][4])

        # prints the opt. parameters returned by srasvm
        print '. srasvm opt params'
        for fpt_idx in xrange(len(self.fp_thresholds)):
            print '. . fpt {0}, m: {1}, k: {2}, p: {3}'.format(
                self.fp_thresholds[fpt_idx],
                self.margin_factors[srasvm_results["opt_m"][fpt_idx]],
                self.knn_values[srasvm_results["opt_k"][fpt_idx]],
                '%.3f' % srasvm_results["test_rates"][fpt_idx][4])

        # prints the opt. parameters returned by srasvm-os
        print '. srasvm-os opt params'
        for fpt_idx in xrange(len(self.fp_thresholds)):
            print '. . fpt {0}, m: {1}, k: {2}, p: {3}'.format(
                self.fp_thresholds[fpt_idx],
                self.margin_factors[srasvm_os_results["opt_m"][fpt_idx]],
                self.knn_values[srasvm_os_results["opt_k"][fpt_idx]],
                '%.3f' % srasvm_os_results["test_rates"][fpt_idx][4])

        # ends the timer
        sys.stdout.write('. total time to compute rasvm rates')
        self.end_timer(time_ini)

        # --- RASVM-SV RATES -------------------------------------------------

        rasvm_sv_dist = 1  # rasvm-sv, only the support vectors to knn

        # starts the timer
        time_ini = self.start_timer('')
        rasvm_sv_results, rasvm_os_sv_results,\
            srasvm_sv_results, srasvm_os_sv_results =\
            self.compute_svm_knn_rates(svm_results['dv_train'],
                                       svm_results['dv_test'],
                                       svm_results['pl_train'],
                                       svm_results['pl_test'],
                                       bs_results['opt_bias'],
                                       rasvm_sv_dist)

        # limits the training and testing rates to 10 decimal cases
        for i in xrange(len(self.fp_thresholds)):
            rasvm_sv_results['train_rates'][i] = map(
                functools.partial(round, ndigits=10),
                rasvm_sv_results['train_rates'][i])
            rasvm_sv_results['test_rates'][i] = map(
                functools.partial(round, ndigits=10),
                rasvm_sv_results['test_rates'][i])

            rasvm_os_sv_results['train_rates'][i] = map(
                functools.partial(round, ndigits=10),
                rasvm_os_sv_results['train_rates'][i])
            rasvm_os_sv_results['test_rates'][i] = map(
                functools.partial(round, ndigits=10),
                rasvm_os_sv_results['test_rates'][i])

            srasvm_sv_results['train_rates'][i] = map(
                functools.partial(round, ndigits=10),
                srasvm_sv_results['train_rates'][i])
            srasvm_sv_results['test_rates'][i] = map(
                functools.partial(round, ndigits=10),
                srasvm_sv_results['test_rates'][i])

            srasvm_os_sv_results['train_rates'][i] = map(
                functools.partial(round, ndigits=10),
                srasvm_os_sv_results['train_rates'][i])
            srasvm_os_sv_results['test_rates'][i] = map(
                functools.partial(round, ndigits=10),
                srasvm_os_sv_results['test_rates'][i])

        # prints the opt. parameters returned by rasvm
        print '. rasvm-sv opt params'
        for fpt_idx in xrange(len(self.fp_thresholds)):
            print '. . fpt {0}, m: {1}, k: {2}, p: {3}'.format(
                self.fp_thresholds[fpt_idx],
                self.margin_factors[rasvm_sv_results["opt_m"][fpt_idx]],
                self.knn_values[rasvm_sv_results["opt_k"][fpt_idx]],
                '%.3f' % rasvm_sv_results["test_rates"][fpt_idx][4])

        # prints the opt. parameters returned by rasvm-os
        print '. rasvm-os-sv opt params'
        for fpt_idx in xrange(len(self.fp_thresholds)):
            print '. . fpt {0}, m: {1}, k: {2}, p: {3}'.format(
                self.fp_thresholds[fpt_idx],
                self.margin_factors[rasvm_os_sv_results["opt_m"][fpt_idx]],
                self.knn_values[rasvm_os_sv_results["opt_k"][fpt_idx]],
                '%.3f' % rasvm_os_sv_results["test_rates"][fpt_idx][4])

        # prints the opt. parameters returned by srasvm
        print '. srasvm-sv opt params'
        for fpt_idx in xrange(len(self.fp_thresholds)):
            print '. . fpt {0}, m: {1}, k: {2}, p: {3}'.format(
                self.fp_thresholds[fpt_idx],
                self.margin_factors[srasvm_sv_results["opt_m"][fpt_idx]],
                self.knn_values[srasvm_sv_results["opt_k"][fpt_idx]],
                '%.3f' % srasvm_sv_results["test_rates"][fpt_idx][4])

        # prints the opt. parameters returned by srasvm-os
        print '. srasvm-os-sv opt params'
        for fpt_idx in xrange(len(self.fp_thresholds)):
            print '. . fpt {0}, m: {1}, k: {2}, p: {3}'.format(
                self.fp_thresholds[fpt_idx],
                self.margin_factors[srasvm_os_sv_results["opt_m"][fpt_idx]],
                self.knn_values[srasvm_os_sv_results["opt_k"][fpt_idx]],
                '%.3f' % srasvm_os_sv_results["test_rates"][fpt_idx][4])

        # ends the timer
        sys.stdout.write('. total time to compute rasvm-sv rates')
        self.end_timer(time_ini)

        return {
            'svm': svm_results,
            'rasvm': rasvm_results,
            'rasvm_os': rasvm_os_results,
            'srasvm': srasvm_results,
            'srasvm_os': srasvm_os_results,
            'rasvm_sv': rasvm_sv_results,
            'rasvm_os_sv': rasvm_os_sv_results,
            'srasvm_sv': srasvm_sv_results,
            'srasvm_os_sv': srasvm_os_sv_results
        }

    # compute bias-shift rates
    def compute_bias_shift_rates(self, dv_train, dv_test, svm_rates_train):
        # --- gets svm roc points ------------------------------------------

        # gets the roc points (variating the bias)
        train_roc_points, train_roc_bias, train_roc_aoc =\
            compute_roc(dv_train, self.train_y)
        test_roc_points, test_roc_bias, test_roc_aoc =\
            compute_roc(dv_test, self.test_y)

        # --- gets the optimized bias --------------------------------------

        # starts the timer
        opt_bias = []
        for fpt in self.fp_thresholds:
            opt_bias.append(
                get_opt_bias(train_roc_points, train_roc_bias, self.train_pos,
                             self.train_neg, fpt, svm_rates_train, self.bias,
                             self.norm_w))

        # --- computes bias-shift rates ------------------------------------

        # gets the predicted labels with shift-bias (for each fp-threshold)
        pl_test_shift_bias = [get_pl_shift_bias(
            dv_test, b, self.bias, self.norm_w) for b in opt_bias]
        pl_train_shift_bias = [get_pl_shift_bias(
            dv_train, b, self.bias, self.norm_w) for b in opt_bias]

        # computes rates with shift-bias (for each fp-threshold)
        test_pos = self.test_pos if self.test_pos != 0 else 1
        test_neg = self.test_neg if self.test_neg != 0 else 1
        train_pos = self.train_pos if self.train_pos != 0 else 1
        train_neg = self.train_neg if self.train_neg != 0 else 1

        opt_bias_rates_test = []
        opt_bias_rates_train = []
        for i in xrange(len(self.fp_thresholds)):
            m_test = get_measures(pl_test_shift_bias[i], self.test_y)
            p_test = get_performance_measure(m_test[0], m_test[1],
                                             m_test[2], m_test[3])
            nps_test = get_np_score(self.fp_thresholds[i],
                                    m_test[1] / float(test_neg),
                                    m_test[3] / float(test_pos))
            opt_bias_rates_test.append([m_test[0] / float(test_pos),
                                        m_test[1] / float(test_neg),
                                        m_test[2] / float(test_neg),
                                        m_test[3] / float(test_pos),
                                        p_test, nps_test])

            m_train = get_measures(pl_train_shift_bias[i], self.train_y)
            p_train = get_performance_measure(m_train[0], m_train[1],
                                              m_train[2], m_train[3])
            nps_train = get_np_score(self.fp_thresholds[i],
                                     m_train[1] / float(train_neg),
                                     m_train[3] / float(train_pos))
            opt_bias_rates_train.append([m_train[0] / float(train_pos),
                                         m_train[1] / float(train_neg),
                                         m_train[2] / float(train_neg),
                                         m_train[3] / float(train_pos),
                                         p_train, nps_train])
        bs_result = {
            'train_rates': opt_bias_rates_train,
            'test_rates': opt_bias_rates_test,
            'opt_bias': opt_bias
        }

        return bs_result

    # export results
    def export_results(self, r, dyn_split):
        # --- exports svm rates --------------------------------------------

        # exports original svm rate results
        self.export_original_svm_results_text(
            r['svm']['train_rates'], r['svm']['test_rates'], dyn_split)

        # --- exports rasvm rates ----------------------------------------

        # saving knn rate results on human-readable text
        self.export_svm_knn_results_text(
            r['rasvm']['opt_k'], r['rasvm']['opt_m'],
            r['rasvm']['train_rates'], r['rasvm']['test_rates'], 'rasvm-ra',
            dyn_split)

        # saving knn rate results on human-readable text
        self.export_svm_knn_results_text(
            r['rasvm_os']['opt_k'], r['rasvm_os']['opt_m'],
            r['rasvm_os']['train_rates'], r['rasvm_os']['test_rates'],
            'rasvm-osra', dyn_split)

        # saving knn rate results on human-readable text
        self.export_svm_knn_results_text(
            r['srasvm']['opt_k'], r['srasvm']['opt_m'],
            r['srasvm']['train_rates'], r['srasvm']['test_rates'],
            'rasvm-sra', dyn_split)

        # saving knn rate results on human-readable text
        self.export_svm_knn_results_text(
            r['srasvm_os']['opt_k'], r['srasvm_os']['opt_m'],
            r['srasvm_os']['train_rates'], r['srasvm_os']['test_rates'],
            'rasvm-ossra', dyn_split)

        # --- exports rasvm-sv rates -------------------------------------

        # saving knn rate results on human-readable text
        self.export_svm_knn_results_text(
            r['rasvm_sv']['opt_k'], r['rasvm_sv']['opt_m'],
            r['rasvm_sv']['train_rates'], r['rasvm_sv']['test_rates'],
            'rasvmsv-ra', dyn_split)

        # saving knn rate results on human-readable text
        self.export_svm_knn_results_text(
            r['rasvm_os_sv']['opt_k'], r['rasvm_os_sv']['opt_m'],
            r['rasvm_os_sv']['train_rates'], r['rasvm_os_sv']['test_rates'],
            'rasvmsv-osra', dyn_split)

        # saving knn rate results on human-readable text
        self.export_svm_knn_results_text(
            r['srasvm_sv']['opt_k'], r['srasvm_sv']['opt_m'],
            r['srasvm_sv']['train_rates'], r['srasvm_sv']['test_rates'],
            'rasvmsv-sra', dyn_split)

        # saving knn rate results on human-readable text
        self.export_svm_knn_results_text(
            r['srasvm_os_sv']['opt_k'], r['srasvm_os_sv']['opt_m'],
            r['srasvm_os_sv']['train_rates'], r['srasvm_os_sv']['test_rates'],
            'rasvmsv-ossra', dyn_split)

    # starts the timer
    def start_timer(self, s):
        sys.stdout.write(s)
        sys.stdout.flush()
        return time.time()

    # prints the current str without line breaks
    def push_str(self, s):
        # prints the str
        sys.stdout.write(s)
        sys.stdout.flush()

    # ends the timer
    def end_timer(self, time_ini):
        print ' [ %s ]' % format_time(time.time() - time_ini)

    # defines the external processing nodes
    def set_ssh_workers(self, nodes):
        # add nodes to the ssh workers list
        for node in nodes:
            node_name = 'no-' + str(node)
            self.ssh_workers += [node_name] * 10

    # checks the amount of parameters
    def check_params(self):
        if len(sys.argv) <= 1:
            print('Usage: {0} training_file [testing_file] pc'
                  .format(sys.argv[0]))
            raise SystemExit

    # gets the main dataset pathnames
    def get_pathnames(self, pc_idx):
        # formatting dataset pathnames
        ds_path = os.path.split(sys.argv[1])[0]
        ds_train_name = os.path.split(sys.argv[1])[1]
        ds_test_name = None
        if len(sys.argv) > 2:
            if sys.argv[2] != '-p':
                ds_test_name = os.path.split(sys.argv[2])[1]
            else:
                self.ds_permuts = int(sys.argv[3])

        # setting output directory path
        output_dir = 'results/'
        output_dir += os.path.split(ds_path + '/' + ds_train_name)[1]
        output_dir = output_dir.split('.train')[0].split('_train')[0]\
                               .split('.scale')[0].split('_scale')[0]\
                               .split('_normalized')[0]
        if self.pos_classes != [1]:
            output_dir += '.class-' + str(self.pos_classes[pc_idx])
        output_dir += '/'

        if ds_test_name is not None:
            output_dir_data = output_dir + 'data/'
            output_dir_data_tmp = '/tmp/data-tmp/'
        else:
            output_dir_data = '/tmp/data-tmp/'
            output_dir_data_tmp = '/tmp/data-tmp/'
        output_dir_res = output_dir + 'results/'
        if not os.path.exists(output_dir_data):
            os.makedirs(output_dir_data)
        if not os.path.exists(output_dir_res):
            os.makedirs(output_dir_res)

        # adding an unique directory id to the output path
        if len(sys.argv) > 3:
            dir_idx = sys.argv[3]
        else:
            res_files = [int(item) for item in os.listdir(output_dir_res)]
            if 0 in res_files:
                os.rename(output_dir_res + '0',
                          output_dir_res + str(max(res_files) + 1))
        dir_idx = 0
        output_dir_res += str(dir_idx) + '/'
        if not os.path.exists(output_dir_res):
            os.makedirs(output_dir_res)

        return {'ds_path': ds_path,
                'ds_test_name': ds_test_name,
                'ds_train_name': ds_train_name,
                'out_path_res': output_dir_res,
                'out_path_data': output_dir_data,
                'out_path_data_tmp': output_dir_data_tmp}

    # sets the stadard pathnames
    def set_std_pathnames(self, pc_idx):
        # sets the dataset input path
        self.ds_train_path =\
            self.pn['ds_path'] + '/' + self.pn['ds_train_name']

        if self.pn['ds_test_name'] is not None:
            self.ds_test_path =\
                self.pn['ds_path'] + '/' + self.pn['ds_test_name']
        else:
            self.ds_test_path = None

        # sets the output filenames
        self.ds_train_out_path_data =\
            self.pn['out_path_data'] + self.pn['ds_train_name']
        self.ds_train_out_path_res =\
            self.pn['out_path_res'] + self.pn['ds_train_name']

        if self.pn['ds_test_name'] is not None:
            self.ds_test_out_path_data =\
                self.pn['out_path_data'] + self.pn['ds_test_name']
            self.ds_test_out_path_res =\
                self.pn['out_path_res'] + self.pn['ds_test_name']
        else:
            self.ds_test_out_path_data = None
            self.ds_test_out_path_res = None

        if self.pos_classes != [1]:
            sufix = '.class-' + str(self.pos_classes[pc_idx])
            self.ds_train_out_path_data += sufix
            self.ds_train_out_path_res += sufix

            if self.ds_test_out_path_data is not None:
                self.ds_test_out_path_data += sufix
            if self.ds_test_out_path_res is not None:
                self.ds_test_out_path_res += sufix

        # sets the scaled dataset output path
        self.ds_train_scale_path = self.ds_train_out_path_data + '.scale'
        self.ds_test_scale_path = self.ds_test_out_path_data + '.scale'\
            if self.ds_test_out_path_data is not None else None

        self.dist_path = self.pn['out_path_data'] + 'dist/'
        self.dist_out_path = self.dist_path + self.pn['ds_train_name']
        if not os.path.exists(self.dist_path):
            os.makedirs(self.dist_path)

        self.dist_path_tmp = self.pn['out_path_data_tmp'] + 'dist/'
        self.dist_out_path_tmp = self.dist_path_tmp + self.pn['ds_train_name']
        if not os.path.exists(self.dist_path_tmp):
            os.makedirs(self.dist_path_tmp)

    # get random subsets lines to cross validation
    def stratified_subsets_lines(self, labels, fold):
        subsets = []
        lines = [i for i in xrange(len(labels))]
        r_labels = labels
        r_lines = lines
        sub_size = math.floor(len(r_lines) / fold)
        for i in xrange(fold - 1):
            random.seed()
            sub_lines = stratified_lines_selection(r_lines, r_labels,
                                                   int(sub_size))
            subsets.append(sub_lines)
            r_labels = [l for l_idx, l in enumerate(r_labels)
                        if r_lines[l_idx] not in sub_lines]
            r_lines = [l for l in r_lines if l not in sub_lines]
        subsets.append(r_lines)
        return subsets

    # compute original svm rates
    def compute_svm_rates(self, model, verbose, svm_type):
        if verbose:
            # starts the timer
            time_ini = self.start_timer('. computing decision values ...')
        # gets the predicted labels (pl) and the decision values (dv)
        pl_test, dv_test, rt_test = svm_predict(
            self.test_x, self.test_y, model)
        pl_train, dv_train, rt_train = svm_predict(
            self.train_x, self.train_y, model)
        if verbose:
            # ends the timer
            self.end_timer(time_ini)

        if verbose:
            # starts the timer
            time_ini = self.start_timer('. computing rates ...')
        # computes svm rates from train data
        tp, fp, tn, fn = get_measures(pl_train, self.train_y)
        pos = self.train_pos if self.train_pos != 0 else 1
        neg = self.train_neg if self.train_neg != 0 else 1
        rates_train = [tp / float(pos), fp / float(neg), tn / float(neg),
                       fn / float(pos)]
        rates_train.append(get_performance_measure(tp, fp, tn, fn))
        # computes svm rates from test data
        tp, fp, tn, fn = get_measures(pl_test, self.test_y)
        pos = self.test_pos if self.test_pos != 0 else 1
        neg = self.test_neg if self.test_neg != 0 else 1
        rates_test = [tp / float(pos), fp / float(neg), tn / float(neg),
                      fn / float(pos)]
        rates_test.append(get_performance_measure(tp, fp, tn, fn))

        if verbose:
            # ends the timer
            self.end_timer(time_ini)

        if verbose:
            # prints the svm predict accuracy results
            print '. . accuracy (train data): ' + rt_train
            print '. . false positive (train data): %0.4f' % rates_train[1]
            print '. . accuracy (test data): ' + rt_test
            print '. . false positive (test data): %0.4f' % rates_test[1]

        result = {
            'train_rates': rates_train,
            'test_rates': rates_test,
            'pl_train': pl_train,
            'dv_train': dv_train,
            'pl_test': pl_test,
            'dv_test': dv_test
        }

        return result

    def createdb(self, c):
        c.execute('''CREATE TABLE neighbors_train
                     (id_orig INTEGER NOT NULL,
                      id_dest INTEGER NOT NULL,
                      max_dist INTEGER NOT NULL,
                      pos integer INTEGER NOT NULL,
                      PRIMARY KEY (id_orig, id_dest, max_dist))''')
        c.execute('''CREATE TABLE neighbors_test
                     (id_orig INTEGER NOT NULL,
                      id_dest INTEGER NOT NULL,
                      max_dist INTEGER NOT NULL,
                      pos integer INTEGER NOT NULL,
                      PRIMARY KEY (id_orig, id_dest, max_dist))''')

    def opendb(self, db):
        db_exists = True if os.path.isfile(db) else False
        conn = lite.connect(db)
        cur = conn.cursor()
        if not db_exists:
            self.createdb(cur)
            conn.commit()
        return [conn, cur]

    # compute rasvm rates
    def compute_svm_knn_rates(self, dv_train, dv_test, pl_train, pl_test,
                              opt_bias, distance_threshold):

        if distance_threshold == 0:
            strategy = 'rasvm'
        elif distance_threshold == 1:
            strategy = 'rasvm-sv'
        else:
            strategy = 'rasvm-d' + str(distance_threshold)

        print '\n> COMPUTING RATE RESULTS FOR ' + strategy.upper()

        db = '%s.sqlite' % self.dist_out_path_tmp
        conn, cur = self.opendb(db)

        # --- computes train-to-train distances ----------------------------

        # starts the timer
        time_ini = self.start_timer(
            '. computing train-to-train distances ...')

        # consider only training points that are inside
        # the distance threshold area
        if distance_threshold == 0:
            sel_train_rows = self.train_lines
        else:
            sv_rows_pos = [i for i in self.train_lines
                           if self.train_y[i] == +1 and
                           dv_train[i] <= distance_threshold]
            sv_rows_neg = [i for i in self.train_lines
                           if self.train_y[i] == -1 and
                           dv_train[i] >= -distance_threshold]
            sel_train_rows = sv_rows_pos + sv_rows_neg
            sel_train_rows.sort()

        sel_train_rows_path = self.dist_out_path + '.train.'
        sel_train_rows_path += str(distance_threshold) + '.rows'
        f = open(sel_train_rows_path, 'w')
        f.write(','.join([str(i) for i in sel_train_rows]))
        f.close()

        # checks if all the required train distances are already computed
        # otherwise, compute everything again
        cur.execute('SELECT count(DISTINCT id_orig) FROM neighbors_train ' +
                    'WHERE max_dist = ?', (distance_threshold,))
        total_rows = cur.fetchone()[0]

        if total_rows == self.total_train:
            # ends the timer
            self.end_timer(time_ini)
            print '. . train-to-train distances were already computed'
        else:
            jobs = self.train_lines
            job_queue = Queue.Queue(0)
            result_queue = Queue.Queue(0)
            for j in jobs:
                job_queue.put((self.train_x,
                               self.train_x, j,
                               sel_train_rows, self.opt_gamma, db,
                               distance_threshold, 'train',
                               self.closest_points_depth))

            # hack the queue to become a stack --
            # this is important when some thread
            # failed and re-put a job. It we still
            # use FIFO, the job will be put
            # into the end of the queue, and the graph
            # will only be updated in the end
            job_queue._put = job_queue.queue.appendleft

            # fire local workers
            for i in xrange(2 * self.nr_local_worker):
                LocalWorker('local', job_queue, result_queue).start()

            # gather results
            done_jobs = {}
            for j in jobs:
                while (j) not in done_jobs:
                    (worker, row, dist_type, neighbors) = result_queue.get()
                    sql = """INSERT INTO neighbors_%s
                            (id_orig, id_dest, max_dist, pos)
                            VALUES (?, ?, ?, ?)""" % dist_type
                    cur.executemany(sql, neighbors)
                    done_jobs[(j)] = 1

            job_queue.put((WorkerStopToken, None, None, None, None, None,
                           None, None, None))

            conn.commit()

            # ends the timer
            self.end_timer(time_ini)

        # --- gets the optmized rasvm params -----------------------------

        # starts the timer
        time_ini = self.start_timer(
            '. computing %s cross-validation rates' % strategy)
        # computes rates in all combinations of parameters
        svmknn_cross_rates, svmknn_cross_rates_one_sided =\
            compute_svmknn_cross_rates(
                self.train_x, self.train_y, self.subsets_lines, sel_train_rows,
                db, distance_threshold, 'train', self.svmext_fold,
                self.knn_values, self.margin_factors, self.margin_size,
                self.model, self.bias, self.bias, self.norm_w,
                self.pos_min_rate, False)
        # ends the timer
        self.end_timer(time_ini)

        # starts the timer
        time_ini = self.start_timer(
            '. computing %s + margin shift cross-validation rates' % strategy)
        # computes rates in all combinations of parameters with
        # bias-shift and margin-shift
        svmknn_cross_rates_bshift_mshift = []
        svmknn_cross_rates_bshift_mshift_os = []
        for b in opt_bias:
            rates, rates_one_sided = compute_svmknn_cross_rates(
                self.train_x, self.train_y, self.subsets_lines, sel_train_rows,
                db, distance_threshold, 'train', self.svmext_fold,
                self.knn_values, self.margin_factors, self.margin_size,
                self.model, b, self.bias, self.norm_w, self.pos_min_rate, True)
            svmknn_cross_rates_bshift_mshift.append(rates)
            svmknn_cross_rates_bshift_mshift_os.append(rates_one_sided)
        # ends the timer
        self.end_timer(time_ini)

        # starts the timer
        time_ini = self.start_timer(
            '. getting optimized parameters ...')
        # gets best svmknn params
        best_m, best_k = [], []
        best_m_one_sided, best_k_one_sided = [], []
        for fpt in self.fp_thresholds:
            # two sided
            m, k = get_best_svmknn_params(
                svmknn_cross_rates, fpt, self.svmext_fold,
                self.margin_factors, self.knn_values, False)
            best_m.append(m)
            best_k.append(k)
            # one sided
            m, k = get_best_svmknn_params(
                svmknn_cross_rates_one_sided, fpt, self.svmext_fold,
                self.margin_factors, self.knn_values, False)
            best_m_one_sided.append(m)
            best_k_one_sided.append(k)
        # gets the best params with margin-shift
        best_m_bshift_mshift, best_k_bshift_mshift = [], []
        best_m_bshift_mshift_one_sided, best_k_bshift_mshift_one_sided = [], []
        for fpt_idx in xrange(len(self.fp_thresholds)):
            # two sided
            m, k = get_best_svmknn_params(
                svmknn_cross_rates_bshift_mshift[fpt_idx],
                self.fp_thresholds[fpt_idx], self.svmext_fold,
                self.margin_factors, self.knn_values, True)
            best_m_bshift_mshift.append(m)
            best_k_bshift_mshift.append(k)
            # one sided
            m, k = get_best_svmknn_params(
                svmknn_cross_rates_bshift_mshift_os[fpt_idx],
                self.fp_thresholds[fpt_idx], self.svmext_fold,
                self.margin_factors, self.knn_values, True)
            best_m_bshift_mshift_one_sided.append(m)
            best_k_bshift_mshift_one_sided.append(k)
        # ends the timer
        self.end_timer(time_ini)

        # --- computes rasvm rates on val. data---------------------------

        # starts the timer
        time_ini = self.start_timer(
            '. computing rates on validation data ...')
        best_val_rates =\
            [[0 for i in xrange(11)] for j in xrange(len(self.fp_thresholds))]
        best_val_rates_one_sided =\
            [[0 for i in xrange(11)] for j in xrange(len(self.fp_thresholds))]
        best_val_rates_bshift_mshift =\
            [[0 for i in xrange(11)] for j in xrange(len(self.fp_thresholds))]
        best_val_rates_bshift_mshift_one_sided =\
            [[0 for i in xrange(11)] for j in xrange(len(self.fp_thresholds))]

        for t in xrange(len(self.fp_thresholds)):
            for f in xrange(self.svmext_fold):
                for i in xrange(len(best_val_rates[t])):
                    # we will set the np-score later
                    if i == 5:
                        continue
                    cr_i = i if i < 5 else i - 1
                    # gets rates for two-sided rasvm
                    m, k = best_m[t], best_k[t]
                    best_val_rates[t][i] +=\
                        svmknn_cross_rates[m][k][f][cr_i]
                    # gets rates for one-sided rasvm
                    m, k = best_m_one_sided[t], best_k_one_sided[t]
                    best_val_rates_one_sided[t][i] +=\
                        svmknn_cross_rates_one_sided[m][k][f][cr_i]
                    # gets rates for two-sided rasvm-mshift
                    m, k = best_m_bshift_mshift[t], best_k_bshift_mshift[t]
                    best_val_rates_bshift_mshift[t][i] +=\
                        svmknn_cross_rates_bshift_mshift[t][m][k][f][cr_i]
                    # gets rates for one-sided rasvm-mshift
                    m, k = best_m_bshift_mshift_one_sided[t],\
                        best_k_bshift_mshift_one_sided[t]
                    best_val_rates_bshift_mshift_one_sided[t][i] +=\
                        svmknn_cross_rates_bshift_mshift_os[t][m][k][f][cr_i]
            # averaging fold results
            for i in [0, 1, 2, 3, 4, 8]:
                best_val_rates[t][i] =\
                    best_val_rates[t][i] /\
                    float(self.svmext_fold)
                best_val_rates_one_sided[t][i] =\
                    best_val_rates_one_sided[t][i] /\
                    float(self.svmext_fold)
                best_val_rates_bshift_mshift[t][i] =\
                    best_val_rates_bshift_mshift[t][i] /\
                    float(self.svmext_fold)
                best_val_rates_bshift_mshift_one_sided[t][i] =\
                    best_val_rates_bshift_mshift_one_sided[t][i] /\
                    float(self.svmext_fold)

            # setting np-score
            best_val_rates[t][5] =\
                get_np_score(self.fp_thresholds[t], best_val_rates[t][1],
                             best_val_rates[t][3])
            best_val_rates_one_sided[t][5] =\
                get_np_score(self.fp_thresholds[t],
                             best_val_rates_one_sided[t][1],
                             best_val_rates_one_sided[t][3])
            best_val_rates_bshift_mshift[t][5] =\
                get_np_score(self.fp_thresholds[t],
                             best_val_rates_bshift_mshift[t][1],
                             best_val_rates_bshift_mshift[t][3])
            best_val_rates_bshift_mshift_one_sided[t][5] =\
                get_np_score(self.fp_thresholds[t],
                             best_val_rates_bshift_mshift_one_sided[t][1],
                             best_val_rates_bshift_mshift_one_sided[t][3])

        # ends the timer
        self.end_timer(time_ini)

        # --- computes test-to-train distances -----------------------------

        # starts the timer
        time_ini = self.start_timer(
            '. computing test-to-train distances ...')

        cur.execute('SELECT count(DISTINCT id_orig) FROM neighbors_test ' +
                    'WHERE max_dist = ?', (distance_threshold,))
        total_rows = cur.fetchone()[0]

        if total_rows == self.total_test:
            # ends the timer
            self.end_timer(time_ini)
            print '. . test-to-train distances were already computed'
        else:
            jobs = self.test_lines
            job_queue = Queue.Queue(0)
            result_queue = Queue.Queue(0)
            for j in jobs:
                job_queue.put((
                    self.test_x, self.train_x, j,
                    sel_train_rows, self.opt_gamma, db,
                    distance_threshold, 'test', self.closest_points_depth))

            # hack the queue to become a stack --
            # this is important when some thread
            # failed and re-put a job. It we still
            # use FIFO, the job will be put
            # into the end of the queue, and the graph
            # will only be updated in the end
            job_queue._put = job_queue.queue.appendleft

            # fire local workers
            for i in xrange(2 * self.nr_local_worker):
                LocalWorker('local', job_queue, result_queue).start()

            # gather results
            done_jobs = {}
            for j in jobs:
                while (j) not in done_jobs:
                    (worker, row, dist_type, neighbors) = result_queue.get()
                    sql = """INSERT INTO neighbors_%s
                            (id_orig, id_dest, max_dist, pos)
                            VALUES (?, ?, ?, ?)""" % dist_type
                    cur.executemany(sql, neighbors)
                    done_jobs[(j)] = 1

            job_queue.put((WorkerStopToken, None, None, None, None, None,
                           None, None, None))

            conn.commit()

            # ends the timer
            self.end_timer(time_ini)

        cur.close()
        conn.close()

        # --- computes rasvm rates on test data --------------------------

        # starts the timer
        time_ini = self.start_timer('. computing rates on test data ...')

        # gets rasvm test rates
        best_test_rates = [compute_svmknn_rates(
            self.train_y, self.test_y, pl_test, dv_test, self.test_lines, db,
            distance_threshold, 'test', best_m[fpt_idx], best_k[fpt_idx],
            self.knn_values, self.margin_factors, self.margin_size,
            self.norm_w, self.pos_min_rate, self.fp_thresholds[fpt_idx], False,
            False, False)
            for fpt_idx in xrange(len(self.fp_thresholds))]
        best_test_rates_one_sided = [compute_svmknn_rates(
            self.train_y, self.test_y, pl_test, dv_test, self.test_lines, db,
            distance_threshold, 'test', best_m_one_sided[fpt_idx],
            best_k_one_sided[fpt_idx], self.knn_values, self.margin_factors,
            self.margin_size, self.norm_w, self.pos_min_rate,
            self.fp_thresholds[fpt_idx], False, False, True)
            for fpt_idx in xrange(len(self.fp_thresholds))]

        # gets rasvm test rates with bias-shift and margin-shift
        pl_test_bshift = [get_pl_shift_bias(
            dv_test, b, self.bias, self.norm_w)
            for b in opt_bias]
        dv_test_mshift = [get_dv_shift_bias(
            dv_test, b, self.bias, self.norm_w)
            for b in opt_bias]
        best_test_rates_bshift_mshift = [compute_svmknn_rates(
            self.train_y, self.test_y, pl_test_bshift[fpt_idx],
            dv_test_mshift[fpt_idx], self.test_lines, db, distance_threshold,
            'test', best_m_bshift_mshift[fpt_idx],
            best_k_bshift_mshift[fpt_idx], self.knn_values,
            self.margin_factors, self.margin_size, self.norm_w,
            self.pos_min_rate, self.fp_thresholds[fpt_idx], True, True, False)
            for fpt_idx in xrange(len(self.fp_thresholds))]
        best_test_rates_bshift_mshift_one_sided = [compute_svmknn_rates(
            self.train_y, self.test_y, pl_test_bshift[fpt_idx],
            dv_test_mshift[fpt_idx], self.test_lines, db, distance_threshold,
            'test', best_m_bshift_mshift_one_sided[fpt_idx],
            best_k_bshift_mshift_one_sided[fpt_idx], self.knn_values,
            self.margin_factors, self.margin_size, self.norm_w,
            self.pos_min_rate, self.fp_thresholds[fpt_idx], True, True, True)
            for fpt_idx in xrange(len(self.fp_thresholds))]

        # ends the timer
        self.end_timer(time_ini)

        if self.dist_out_path_tmp != self.dist_out_path:
            shutil.copy('%s.sqlite' % self.dist_out_path_tmp,
                        '%s.sqlite' % self.dist_out_path)

        rasvm = {
            'train_rates': best_val_rates,
            'test_rates': best_test_rates,
            'opt_k': best_k,
            'opt_m': best_m
        }

        rasvm_os = {
            'train_rates': best_val_rates_one_sided,
            'test_rates': best_test_rates_one_sided,
            'opt_k': best_k_one_sided,
            'opt_m': best_m_one_sided
        }

        srasvm = {
            'train_rates': best_val_rates_bshift_mshift,
            'test_rates': best_test_rates_bshift_mshift,
            'opt_k': best_k_bshift_mshift,
            'opt_m': best_m_bshift_mshift
        }

        srasvm_os = {
            'train_rates': best_val_rates_bshift_mshift_one_sided,
            'test_rates': best_test_rates_bshift_mshift_one_sided,
            'opt_k': best_k_bshift_mshift_one_sided,
            'opt_m': best_m_bshift_mshift_one_sided
        }

        return rasvm, rasvm_os, srasvm, srasvm_os

    # exports original svm results in human-readable text
    def export_original_svm_results_text(self, train_rates, test_rates,
                                         dyn_split):
        # open the file in write mode
        f = open(self.ds_test_out_path_res + '-svm.result', 'w')

        # if we didn't used 5x2 cross-validation
        if not dyn_split:
            s = 'TP: %.3f | FP: %.3f | TN: %.3f | FN: %.3f | F1: %.3f\n'

            f.write('Train data\n' + s %
                    (train_rates[0], train_rates[1], train_rates[2],
                     train_rates[3], train_rates[4]))

            f.write('Test data\n' + s %
                    (test_rates[0], test_rates[1], test_rates[2],
                     test_rates[3], test_rates[4]))

        # if we used 5x2 cross validation (show std deviation)
        else:
            s = 'TP: %.3f (%.2f) | FP: %.3f (%.2f) | TN: %.3f (%.2f) | ' +\
                'FN: %.3f (%.2f) | F1: %.3f (%.2f)\n'

            f.write('Train data\n' + s %
                    (train_rates[0][0], train_rates[0][1],
                     train_rates[1][0], train_rates[1][1],
                     train_rates[2][0], train_rates[2][1],
                     train_rates[3][0], train_rates[3][1],
                     train_rates[4][0], train_rates[4][1]))

            f.write('Test data\n' + s %
                    (test_rates[0][0], test_rates[0][1],
                     test_rates[1][0], test_rates[1][1],
                     test_rates[2][0], test_rates[2][1],
                     test_rates[3][0], test_rates[3][1],
                     test_rates[4][0], test_rates[4][1]))

        f.close()

    # exports svm knn results in human-readable text
    def export_svm_knn_results_text(self, best_k, best_m, best_val_rates,
                                    test_knn_rates, result_type, dyn_split):
        # open the file in write mode
        path = self.ds_test_out_path_res + '-' + result_type + '.result'
        f = open(path, 'w')

        # if we didn't used 5x2 cross-validation
        if not dyn_split:
            s = 'FPT: %.3f | MF: %.3f | K: %2.0f | TP: %.3f | FP: %.3f |' +\
                ' TN: %.3f | FN: %.3f | F1: %.3f | NP: %.3f |' +\
                ' Hits: %2.0f/%2.0f (%3.2f%%) |' +\
                ' Misses (FP/FN): %2.0f/%2.0f\n'

            f.write('Train data:\n')
            for i in xrange(len(self.fp_thresholds)):
                f.write(s %
                        (self.fp_thresholds[i], self.margin_factors[best_m[i]],
                         self.knn_values[best_k[i]], best_val_rates[i][0],
                         best_val_rates[i][1], best_val_rates[i][2],
                         best_val_rates[i][3], best_val_rates[i][4],
                         best_val_rates[i][5], best_val_rates[i][6],
                         best_val_rates[i][7], best_val_rates[i][8],
                         best_val_rates[i][9], best_val_rates[i][10]))

            f.write('Test data:\n')
            for i in xrange(len(self.fp_thresholds)):
                f.write(s %
                        (self.fp_thresholds[i], self.margin_factors[best_m[i]],
                         self.knn_values[best_k[i]], test_knn_rates[i][0],
                         test_knn_rates[i][1], test_knn_rates[i][2],
                         test_knn_rates[i][3], test_knn_rates[i][4],
                         test_knn_rates[i][5], test_knn_rates[i][6],
                         test_knn_rates[i][7], test_knn_rates[i][8],
                         test_knn_rates[i][9], test_knn_rates[i][10]))

        # if we used 5x2 cross validation (show std deviation)
        else:
            s = 'FPT: %.3f | TP: %.3f (%.2f) | ' +\
                'FP: %.3f (%.2f) | TN: %.3f (%.2f) | ' +\
                'FN: %.3f (%.2f) | F1: %.3f (%.2f) | ' +\
                'NP: %.3f (%.2f) | Hits: %2.0f/%2.0f (%3.2f%%) | ' +\
                'Misses (FP/FN): %2.0f/%2.0f\n'

            f.write('Train data:\n')
            for i in xrange(len(self.fp_thresholds)):
                f.write(s %
                        (self.fp_thresholds[i],
                         best_val_rates[i][0][0], best_val_rates[i][0][1],
                         best_val_rates[i][1][0], best_val_rates[i][1][1],
                         best_val_rates[i][2][0], best_val_rates[i][2][1],
                         best_val_rates[i][3][0], best_val_rates[i][3][1],
                         best_val_rates[i][4][0], best_val_rates[i][4][1],
                         best_val_rates[i][5][0], best_val_rates[i][5][1],
                         best_val_rates[i][6][0], best_val_rates[i][7][0],
                         best_val_rates[i][8][0], best_val_rates[i][9][0],
                         best_val_rates[i][10][0]))

            f.write('Test data:\n')
            for i in xrange(len(self.fp_thresholds)):
                f.write(s %
                        (self.fp_thresholds[i],
                         test_knn_rates[i][0][0], test_knn_rates[i][0][1],
                         test_knn_rates[i][1][0], test_knn_rates[i][1][1],
                         test_knn_rates[i][2][0], test_knn_rates[i][2][1],
                         test_knn_rates[i][3][0], test_knn_rates[i][3][1],
                         test_knn_rates[i][4][0], test_knn_rates[i][4][1],
                         test_knn_rates[i][5][0], test_knn_rates[i][5][1],
                         test_knn_rates[i][6][0], test_knn_rates[i][7][0],
                         test_knn_rates[i][8][0], test_knn_rates[i][9][0],
                         test_knn_rates[i][10][0]))

        f.close()


# Worker Stop Token
class WorkerStopToken:  # used to notify the worker to stop
    pass


# Worker
class Worker(Thread):
    def __init__(self, name, job_queue, result_queue):
        Thread.__init__(self)
        self.name = name
        self.job_queue = job_queue
        self.result_queue = result_queue

    def run(self):
        while True:
            (df_x, dt_x, row, sel_rows, g, db, dist, dist_type, depth) =\
                self.job_queue.get()
            if df_x is WorkerStopToken:
                self.job_queue.put((df_x, dt_x, row, sel_rows, g, db, dist,
                                    dist_type, depth))
                break
            try:
                dists = self.run_one(df_x, dt_x, row, sel_rows, g, db, dist,
                                     dist_type, depth)
                neighbors = []
                for i, rid in enumerate(dists):
                    id_orig = row
                    id_dest = rid
                    pos = i
                    neighbors.append((id_orig, id_dest, dist, pos,))
            except:
                # we failed, let others do that and we just quit
                traceback.print_exception(
                    sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2])
                self.job_queue.put((df_x, dt_x, row, sel_rows, g, db, dist,
                                    dist_type, depth))
                print('worker {0} quit.'.format(self.name))
                break
            else:
                self.result_queue.put((self.name, row, dist_type, neighbors))


# Local Worker
class LocalWorker(Worker):
    def run_one(self, df_x, dt_x, row, sel_rows, g, db, dist, dist_type,
                depth):
        distances = []
        for r in sel_rows:
            d = kernel_rbf(df_x[row], dt_x[r], float(g))
            distances.append(d)
        distances_sorted_idx = sorted(range(len(distances)),
                                      reverse=True,
                                      key=lambda k: distances[k])
        return [sel_rows[i] for i in distances_sorted_idx[0:int(depth)]]


main = main()
