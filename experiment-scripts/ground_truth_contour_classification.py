"""Runs all ground truth contour classification experiments"""
from __future__ import print_function

import argparse
import matplotlib.pyplot as plt
import medleydb as mdb
from medleydb.mix import VOCALS
from medleydb import download
import motif
import numpy as np
import os

import utils
from utils import save_to_json, save_classifier


def fit_model(X_train, Y_train, X_test, Y_test, compute_prob=True):

    clf = motif.contour_classifiers.RandomForest()
    clf.fit(X_train, Y_train)

    if compute_prob:
        Y_train_pred_prob = clf.predict(X_train)
        Y_test_pred_prob = clf.predict(X_test)
    else:
        Y_train_pred_prob = None
        Y_test_pred_prob = None

    Y_train_pred = clf.predict_discrete_label(X_train)
    Y_test_pred = clf.predict_discrete_label(X_test)

    train_score = clf.score(Y_train_pred, Y_train, Y_train_pred_prob)
    test_score = clf.score(Y_test_pred, Y_test, Y_test_pred_prob)

    all_scores = {'train': train_score, 'test': test_score}

    return all_scores, clf


def plot_scores(train_score, test_score, class_labels, clf):
    #TODO: Decide what to do here
    plt.figure(figsize=(12, 12))
    plt.grid('off')

    plt.subplot(2, 1, 1)
    plt.title("training")
    plt.imshow(
        train_score['confusion matrix'], interpolation='none', cmap='hot'
    )
    plt.xticks(np.arange(len(class_labels)), class_labels)
    plt.yticks(np.arange(len(class_labels)), class_labels)
    plt.axis('auto')
    plt.axis('tight')
    plt.colorbar()

    plt.subplot(2, 1, 2)
    plt.title("testing")
    plt.imshow(test_score['confusion matrix'], interpolation='none', cmap='hot')
    plt.xticks(np.arange(len(class_labels)), class_labels)
    plt.yticks(np.arange(len(class_labels)), class_labels)
    plt.axis('auto')
    plt.axis('tight')
    plt.colorbar()

    plt.show()

    plt.figure(figsize=(12, 7))
    y = clf.clf.feature_importances_
    x = np.arange(len(y))
    plt.bar(x, y)
    plt.xticks(x + 0.5, FTR.feature_names, rotation='vertical')
    plt.show()


def main(args):
    save_dir = args.results_path
    save_contour_path = args.save_contour_path
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(save_contour_path):
        os.mkdir(save_contour_path)

    ## Aggregate tracks that will be used in experiment
    all_trackids = utils.get_experiment_trackids()

    ## Create Train-test splits
    data_split = mdb.utils.artist_conditional_split(
        trackid_list=all_trackids, test_size=0.2, num_splits=args.num_splits,
        random_state=7
    )

    # save splits to file
    split_path = os.path.join(save_dir, 'train_test_splits.json')
    split_dict = {i: data_split[i] for i in range(args.num_splits)}
    save_to_json(split_dict, split_path)

    for i in range(args.num_splits):

        this_data_split = data_split[i]
        train_ids = this_data_split['train']
        test_ids = this_data_split['test']

        ## make folder for results for split
        split_dir = os.path.join(save_dir, 'split-{}'.format(i))
        if not os.path.exists(split_dir):
            os.mkdir(split_dir)

        ## Get ground truth contours for each stem
        print("Computing ground truth contours...")
        (train_contours, train_instrument_labels,
         train_component_labels, train_trackid, _) = utils.get_contours(
            train_ids, save_contour_path
        )

        (test_contours, test_instrument_labels,
         test_component_labels, test_trackid, _) = utils.get_contours(
            test_ids, save_contour_path
        )

        ## Run Vocal Non-Vocal Experiment
        print("Running Vocal experiment...")
        try:
            vocal_label_dict = utils.get_vocal_label_dict(
                train_instrument_labels, test_instrument_labels, VOCALS
            )
            vocal_labels = ['non-vocal', 'vocal']
            X_train, Y_train, X_test, Y_test = utils.build_training_testing_set(
                train_contours, train_instrument_labels,
                test_contours, test_instrument_labels,
                vocal_label_dict
            )
            scores_vocal, vocal_clf = fit_model(
                X_train, Y_train, X_test, Y_test
            )

            save_to_json(
                {'labels': vocal_labels},
                os.path.join(split_dir, 'vocal_labels.json')
            )
            save_to_json(
                scores_vocal, os.path.join(split_dir, 'vocal_scores.json')
            )
            save_classifier(
                vocal_clf, os.path.join(split_dir, 'vocal_classifier.pkl')
            )
        except:
            print('[Error] Vocal experiment failed somewhere.')

        ## Run Melody Non-Melody Experiment
        print("Running Melody experiment...")
        try:
            melody_label_dict = utils.get_melody_label_dict(
                train_component_labels, test_component_labels
            )
            melody_labels = ['non-melody', 'melody']
            X_train, Y_train, X_test, Y_test = utils.build_training_testing_set(
                train_contours, train_component_labels,
                test_contours, test_component_labels,
                melody_label_dict
            )
            scores_melody, melody_clf = fit_model(
                X_train, Y_train, X_test, Y_test
            )

            save_to_json(
                {'labels': melody_labels},
                os.path.join(split_dir, 'melody_labels.json')
            )
            save_to_json(
                scores_melody, os.path.join(split_dir, 'melody_scores.json')
            )
            save_classifier(
                melody_clf, os.path.join(split_dir, 'melody_classifier.pkl')
            )
        except:
            print('[Error] Melody experiment failed somewhere.')

        ## Run Bass Non-Bass Experiment
        print("Running Bass experiment...")
        try:
            bass_label_dict = utils.get_bass_label_dict(
                train_component_labels, test_component_labels
            )
            bass_labels = ['non-bass', 'bass']
            X_train, Y_train, X_test, Y_test = utils.build_training_testing_set(
                train_contours, train_component_labels,
                test_contours, test_component_labels,
                bass_label_dict
            )
            scores_bass, bass_clf = fit_model(
                X_train, Y_train, X_test, Y_test
            )

            save_to_json(
                {'labels': bass_labels},
                os.path.join(split_dir, 'bass_labels.json')
            )
            save_to_json(
                scores_bass, os.path.join(split_dir, 'bass_scores.json')
            )
            save_classifier(
                bass_clf, os.path.join(split_dir, 'bass_classifier.pkl')
            )
        except:
            print('[Error] Bass experiment failed somewhere.')

        ## Run Instrument ID experiment
        print("Running Instrument ID experiment...")
        try:
            inst_label_dict, inst_labels = utils.get_inst_label_dict(
                train_instrument_labels, test_instrument_labels
            )
            X_train, Y_train, X_test, Y_test = utils.build_training_testing_set(
                train_contours, train_instrument_labels,
                test_contours, test_instrument_labels,
                inst_label_dict
            )
            scores_inst, inst_clf = fit_model(
                X_train, Y_train, X_test, Y_test, compute_prob=False
            )

            save_to_json(
                {'labels': inst_labels},
                os.path.join(split_dir, 'inst_labels.json')
            )
            save_to_json(
                scores_inst, os.path.join(split_dir, 'inst_scores.json')
            )
            save_classifier(
                inst_clf, os.path.join(split_dir, 'inst_classifier.pkl')
            )
        except:
            print('[Error] Instrument ID experiment failed somewhere.')

        print("    Done with split {}".format(i))

    print("Done!")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Run contour classification experiments.")
    PARSER.add_argument("results_path",
                        type=str,
                        help="Path to save results.")
    PARSER.add_argument("save_contour_path",
                        type=str,
                        help="Path to save/find contours")
    PARSER.add_argument("num_splits",
                        type=int,
                        default=5,
                        help="Number of iterations to run")
    main(PARSER.parse_args())
