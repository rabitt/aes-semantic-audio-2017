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


def main(args):
    save_dir = args.results_path
    save_contour_path = args.save_contour_path
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(save_contour_path):
        os.mkdir(save_contour_path)

    n_harms = args.n_harms
    rm_avg_pitch = bool(args.rm_avg_pitch)

    print("rm_avg_pitch={}".format(rm_avg_pitch))

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
             train_ids, save_contour_path, n_harms
         )

        (test_contours, test_instrument_labels,
         test_component_labels, test_trackid, _) = utils.get_contours(
             test_ids, save_contour_path, n_harms
         )

        # ## Run Vocal Non-Vocal Experiment
        # print("Running Vocal experiment...")
        # try:
        #     vocal_label_dict = utils.get_vocal_label_dict(
        #         train_instrument_labels, test_instrument_labels, VOCALS
        #     )
        #     vocal_labels = ['non-vocal', 'vocal']
        #     X_train, Y_train, X_test, Y_test = utils.build_training_testing_set(
        #         train_contours, train_instrument_labels,
        #         test_contours, test_instrument_labels,
        #         vocal_label_dict, delete_avg_pitch=rm_avg_pitch
        #     )
        #     scores_vocal, vocal_clf = fit_model(
        #         X_train, Y_train, X_test, Y_test
        #     )

        #     save_to_json(
        #         {'labels': vocal_labels},
        #         os.path.join(split_dir, 'vocal_labels.json')
        #     )
        #     save_to_json(
        #         scores_vocal, os.path.join(split_dir, 'vocal_scores.json')
        #     )
        #     save_classifier(
        #         vocal_clf, os.path.join(split_dir, 'vocal_classifier.pkl')
        #     )
        # except:
        #     print('[Error] Vocal experiment failed somewhere.')

        # ## run contour function experiment (melody/bass/other)
        # print("Running Function experiment...")
        # try:
        #     print("    > computing features...")
        #     function_label_dict = utils.get_function_label_dict(
        #         train_component_labels, test_component_labels
        #     )
        #     function_labels = ['other', 'bass', 'melody']
        #     X_train, Y_train, X_test, Y_test = utils.build_training_testing_set(
        #         train_contours, train_component_labels,
        #         test_contours, test_component_labels,
        #         function_label_dict, delete_avg_pitch=rm_avg_pitch
        #     )
        #     print("    > fitting model...")
        #     scores_function, function_clf = fit_model(
        #         X_train, Y_train, X_test, Y_test, compute_prob=False
        #     )

        #     save_to_json(
        #         {'labels': function_labels},
        #         os.path.join(split_dir, 'function_labels.json')
        #     )
        #     save_to_json(
        #         scores_function, os.path.join(split_dir, 'function_scores.json')
        #     )
        #     save_classifier(
        #         function_clf, os.path.join(split_dir, 'function_classifier.pkl')
        #     )
        # except:
        #     print('[Error] Function experiment failed somewhere.')

        # ## Run Melody Non-Melody Experiment
        # print("Running Melody experiment...")
        # try:
        #     melody_label_dict = utils.get_melody_label_dict(
        #         train_component_labels, test_component_labels
        #     )
        #     melody_labels = ['non-melody', 'melody']
        #     X_train, Y_train, X_test, Y_test = utils.build_training_testing_set(
        #         train_contours, train_component_labels,
        #         test_contours, test_component_labels,
        #         melody_label_dict, delete_avg_pitch=rm_avg_pitch
        #     )
        #     scores_melody, melody_clf = fit_model(
        #         X_train, Y_train, X_test, Y_test
        #     )

        #     save_to_json(
        #         {'labels': melody_labels},
        #         os.path.join(split_dir, 'melody_labels.json')
        #     )
        #     save_to_json(
        #         scores_melody, os.path.join(split_dir, 'melody_scores.json')
        #     )
        #     save_classifier(
        #         melody_clf, os.path.join(split_dir, 'melody_classifier.pkl')
        #     )
        # except:
        #     print('[Error] Melody experiment failed somewhere.')

        # ## Run Bass Non-Bass Experiment
        # print("Running Bass experiment...")
        # try:
        #     bass_label_dict = utils.get_bass_label_dict(
        #         train_component_labels, test_component_labels
        #     )
        #     bass_labels = ['non-bass', 'bass']
        #     X_train, Y_train, X_test, Y_test = utils.build_training_testing_set(
        #         train_contours, train_component_labels,
        #         test_contours, test_component_labels,
        #         bass_label_dict, delete_avg_pitch=rm_avg_pitch
        #     )
        #     scores_bass, bass_clf = fit_model(
        #         X_train, Y_train, X_test, Y_test
        #     )

        #     save_to_json(
        #         {'labels': bass_labels},
        #         os.path.join(split_dir, 'bass_labels.json')
        #     )
        #     save_to_json(
        #         scores_bass, os.path.join(split_dir, 'bass_scores.json')
        #     )
        #     save_classifier(
        #         bass_clf, os.path.join(split_dir, 'bass_classifier.pkl')
        #     )
        # except:
        #     print('[Error] Bass experiment failed somewhere.')

        ## Run Vocal gender experiment
        print("Running Vocal type experiment...")
        try:
            voctype_label_dict, voctype_labels = utils.get_vocaltype_label_dict(
                train_instrument_labels, test_instrument_labels,
                ['male singer', 'female singer']
            )
            print(voctype_label_dict)
            X_train, Y_train, X_test, Y_test = utils.build_training_testing_set(
                train_contours, train_instrument_labels,
                test_contours, test_instrument_labels,
                voctype_label_dict, delete_avg_pitch=rm_avg_pitch
            )
            print(Y_test.shape)
            print(set(Y_test))

            scores_voctype, voctype_clf = fit_model(
                X_train, Y_train, X_test, Y_test, compute_prob=True
            )

            save_to_json(
                {'labels': voctype_labels},
                os.path.join(split_dir, 'voctype_labels.json')
            )
            save_to_json(
                scores_voctype, os.path.join(split_dir, 'voctype_scores.json')
            )
            save_classifier(
                voctype_clf, os.path.join(split_dir, 'voctype_classifier.pkl')
            )
        except:
            print('[Error] Vocal type experiment failed somewhere.')

        ## Run 10-class instrument ID experiment
        print("Running Instrument ID experiment...")
        try:
            inst_list = [
                'cello', 'double bass', 'electric bass', 'female singer',
                'male rapper', 'male singer', 'tenor saxophone', 'trumpet',
                'viola', 'violin'
            ]
            inst_label_dict, inst_labels = utils.get_inst_label_dict(
                train_instrument_labels, test_instrument_labels, inst_list
            )
            X_train, Y_train, X_test, Y_test = utils.build_training_testing_set(
                train_contours, train_instrument_labels,
                test_contours, test_instrument_labels,
                inst_label_dict, delete_avg_pitch=rm_avg_pitch
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
    PARSER.add_argument("n_harms",
                        type=int,
                        default=8,
                        help="number of harmonics for salience computation")
    PARSER.add_argument("rm_avg_pitch",
                        type=int,
                        default=0,
                        help="If True removes the feature equivalent to "
                        "average pitch")
    main(PARSER.parse_args())
