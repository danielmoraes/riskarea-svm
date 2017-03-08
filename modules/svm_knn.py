import sys
import sqlite3 as lite
from modules.svm_ext import\
    select_subset, svm_predict, get_pl_shift_bias, get_dv_shift_bias,\
    get_distances, get_measures, get_performance_measure, get_np_score
from modules.util import count_pos_neg, binary_search


# computes svm-knn hit rates
def get_svmknn_hit_rates(labels, svm_labels, svmknn_labels):
    # computes hit rates
    hits, tot_changes, fp_miss, fn_miss = 0, 0, 0, 0
    for i in xrange(len(labels)):
        if int(svm_labels[i]) != int(svmknn_labels[i]):
            tot_changes += 1
            if int(svmknn_labels[i]) == int(labels[i]):
                hits += 1
            else:
                if int(labels[i]) > 0:
                    fn_miss += 1
                else:
                    fp_miss += 1

    hit_rate = 0 if tot_changes == 0\
        else hits / float(tot_changes) * 100
    return hits, tot_changes, hit_rate, fp_miss, fn_miss


# get predicted labels with knn
def get_class_by_knn(closest_points_idx, to_y, k, pos_rate):
    pos = 0
    n = min(k, len(closest_points_idx))
    for i in xrange(n):
        if int(to_y[closest_points_idx[i]]) == 1:
            pos += 1
    return 1 if pos / float(n) >= pos_rate else -1


# computes svm knn rates
def compute_svmknn_cross_rates(ds_x, ds_y, subsets_lines, sel_rows, db, dist,
                               dist_type, fold, knn_values, margin_factors,
                               margin_size, svm_model, shift_bias, svm_bias,
                               norm_w, pos_min_rate, shift_margin_area):

    conn = lite.connect(db)
    cur = conn.cursor()

    rates =\
        [[[None for i in xrange(fold)] for j in xrange(len(knn_values))]
         for k in xrange(len(margin_factors))]

    rates_one_sided =\
        [[[None for i in xrange(fold)] for j in xrange(len(knn_values))]
         for k in xrange(len(margin_factors))]

    for f in xrange(fold):
        sys.stdout.write(' .')
        sys.stdout.flush()

        # mounts validation subset
        sub_train_x, sub_train_y, sub_val_x, sub_val_y = select_subset(
            ds_x, ds_y, subsets_lines[f])

        # --- computing subset pls and dvs ---------------------------------

        # computes subset predicted labels and decision values
        pl_val, dv_val, rt_val = svm_predict(
            sub_val_x, sub_val_y, svm_model)
        if shift_bias != svm_bias:
            # recomputes pl according to the bias-shift hyperplane
            pl_val = get_pl_shift_bias(
                dv_val, shift_bias, svm_bias, norm_w)
            if shift_margin_area:
                # moves margin area of svm according to the
                # bias-shift hyperplane
                dv_val = get_dv_shift_bias(
                    dv_val, shift_bias, svm_bias, norm_w)
        # saves the number os lines from validation data
        total_sub_val = len(sub_val_y)
        # computes the total of pos and neg values from validation data
        sub_val_pos, sub_val_neg = count_pos_neg(sub_val_y)

        # --- computing rates on validation data ---------------------------

        dists = {}
        for m_idx, m in enumerate(margin_factors):
            exp_margin = margin_size * m
            val_margin = [i for i in xrange(total_sub_val)
                          if abs(dv_val[i]) / float(norm_w) < exp_margin
                          and binary_search(sel_rows,
                                            subsets_lines[f][i]) != -1]
            val_margin_one_sided =\
                [i for i in xrange(total_sub_val)
                 if dv_val[i] > 0
                 and abs(dv_val[i]) / float(norm_w) < exp_margin
                 and binary_search(sel_rows, subsets_lines[f][i]) != -1]
            for i in val_margin:
                if i not in dists.keys():
                    dists[i] = get_distances(
                        subsets_lines[f][i], cur, dist, dist_type,
                        subsets_lines[f])
            for k_idx, k in enumerate(knn_values):
                # applying two-sided svm-knn
                pl_val_svmknn = list(pl_val)
                for p in val_margin:
                    c = get_class_by_knn(dists[p], ds_y, k, pos_min_rate)
                    pl_val_svmknn[p] = int(c)
                # applying one-sided svm-knn
                pl_val_svmknn_one_sided = list(pl_val)
                for p in val_margin_one_sided:
                    c = get_class_by_knn(dists[p], ds_y, k, pos_min_rate)
                    pl_val_svmknn_one_sided[p] = int(c)
                # computing two-sided svm-knn rates
                tp, fp, tn, fn = get_measures(pl_val_svmknn, sub_val_y)
                p_measure = get_performance_measure(tp, fp, tn, fn)
                # computing one-sided svm-knn rates
                tp_os, fp_os, tn_os, fn_os = get_measures(
                    pl_val_svmknn_one_sided, sub_val_y)
                p_measure_os = get_performance_measure(
                    tp_os, fp_os, tn_os, fn_os)
                # computing two-sided svm-knn hit rates
                hits, tot_changes, hit_rate, fp_miss, fn_miss =\
                    get_svmknn_hit_rates(sub_val_y, pl_val, pl_val_svmknn)
                # computing one-sided svm-knn hit rates
                hits_os, tot_changes_os, hit_rate_os, fp_miss_os, fn_miss_os =\
                    get_svmknn_hit_rates(sub_val_y, pl_val,
                                         pl_val_svmknn_one_sided)
                # saves the rate results for the current subset
                rates[m_idx][k_idx][f] =\
                    [tp / float(sub_val_pos), fp / float(sub_val_neg),
                     tn / float(sub_val_neg), fn / float(sub_val_pos),
                     p_measure, hits, tot_changes, hit_rate, fp_miss, fn_miss]
                rates_one_sided[m_idx][k_idx][f] =\
                    [tp_os / float(sub_val_pos),
                     fp_os / float(sub_val_neg),
                     tn_os / float(sub_val_neg),
                     fn_os / float(sub_val_pos), p_measure_os, hits_os,
                     tot_changes_os, hit_rate_os, fp_miss_os, fn_miss_os]

    cur.close()
    conn.close()

    return rates, rates_one_sided


# compute svmknn test rates
def compute_svmknn_rates(to_y, ds_y, ds_pl, ds_dv, sel_dist_rows, db, dist,
                         dist_type, best_m, best_k, knn_values, margin_factors,
                         margin_size, norm_w, pos_min_rate, fpt, bias_shift,
                         margin_shift, is_one_sided):

    conn = lite.connect(db)
    cur = conn.cursor()

    # selecting points inside margin
    exp_margin = margin_size * margin_factors[best_m]
    if not is_one_sided:
        inside_margin = [i for i in xrange(len(ds_y))
                         if abs(ds_dv[i]) / norm_w < exp_margin
                         and binary_search(sel_dist_rows, i) != -1]
    else:
        inside_margin = [i for i in xrange(len(ds_y))
                         if ds_dv[i] > 0
                         and abs(ds_dv[i]) / norm_w < exp_margin
                         and binary_search(sel_dist_rows, i) != -1]
    # applying svm-knn
    ds_pl_svmknn = list(ds_pl)
    for p in inside_margin:
        d = get_distances(
            p, cur, dist, dist_type, [])
        c = get_class_by_knn(d, to_y, knn_values[best_k], pos_min_rate)
        ds_pl_svmknn[p] = int(c)
    # computing svm-knn rates
    pos, neg = count_pos_neg(ds_y)
    pos = pos if pos != 0 else 1
    neg = neg if neg != 0 else 1
    tp, fp, tn, fn = get_measures(ds_pl_svmknn, ds_y)
    p_measure = get_performance_measure(tp, fp, tn, fn)
    np_score = get_np_score(fpt, fp / float(neg), fn / float(pos))
    # computing svm-knn hit rates
    hits, tot_changes, hit_rate, fp_miss, fn_miss =\
        get_svmknn_hit_rates(ds_y, ds_pl, ds_pl_svmknn)

    cur.close()
    conn.close()

    return [tp / float(pos), fp / float(neg), tn / float(neg), fn / float(pos),
            p_measure, np_score, hits, tot_changes, hit_rate, fp_miss, fn_miss]


# returns best svm-knn params by threshold
def get_best_svmknn_params(svmknn_cross_rates, fpt, fold, margin_factors,
                           knn_values, is_bias_shift):
    best_m, best_k, best_nps = -1, -1, -1
    for m_idx, m in enumerate(margin_factors):
        for k_idx, k in enumerate(knn_values):
            fold_nps = 0.0
            for f in xrange(fold):
                fpr = svmknn_cross_rates[m_idx][k_idx][f][1]
                fnr = svmknn_cross_rates[m_idx][k_idx][f][3]
                fold_nps += get_np_score(fpt, fpr, fnr)
            if best_nps == -1 or fold_nps < best_nps:
                best_m = m_idx
                best_k = k_idx
                best_nps = fold_nps
    return best_m, best_k
