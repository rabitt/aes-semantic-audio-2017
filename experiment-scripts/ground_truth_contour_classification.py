"""Runs all ground truth contour classification experiments"""
from __future__ import print_function

import argparse
import json
import librosa
import matplotlib.pyplot as plt
import medleydb as mdb
from medleydb.mix import VOCALS
import motif
from motif.core import Contours
from motif import plot
import numpy as np
import os
from pprint import pprint
import scipy


SR = 44100.0/256.0
HOP = (1.0/SR) + 0.0001 
F_THRESH = 1.0/24.0
MDB_MIXES = "/Users/rabitt/Dropbox/MARL/mono"
INTERPOLATORS = {}
FTR = motif.run.get_features_module('bitteli')


class MockEtr(motif.core.ContourExtractor):
    @property
    def audio_samplerate(self):
        return 44100

    @property
    def sample_rate(self):
        return SR

    @property
    def min_contour_len(self):
        return 0.05

    @classmethod
    def get_id(cls):
        return 'mock'

    def compute_contours(self, audio_filepath):
        pass


ETR = MockEtr()


def pitch_to_contours(pitch_data, S_interp, conf_interp, mix_path):
    index = []
    times = []
    freqs = []
    salience = []

    i = 0
    index.append(i)
    times.append(pitch_data[0][0])
    freqs.append(pitch_data[0][1])
    salience.append(S_interp(pitch_data[0][0], pitch_data[0][1]))
    
    for (t_prev, f_prev), (t, f) in zip(pitch_data[:-1], pitch_data[1:]):

        if f == 0 or conf_interp(t) < 0.5:
            i = i + 1
            continue

        if ((t - t_prev > HOP) or f_prev == 0 or
                (np.abs(np.log2(f/f_prev)) > F_THRESH)):
            i = i + 1

        
        index.append(i)
        times.append(t)
        freqs.append(f)
        salience.append(S_interp(t, f))
    
    index, times, freqs, salience = ETR._postprocess_contours(
        np.array(index), np.array(times), np.array(freqs), np.array(salience)
    )

    contour = Contours(index, times, freqs, salience, SR, mix_path)
    return contour


def salience_interpolator(audio_filepath):
    nfft = 2048
    win_length = 2048
    hop_length = 512
    
    y, sr = librosa.load(audio_filepath)
    
    S = librosa.stft(
        y, n_fft=nfft, hop_length=hop_length, win_length=win_length
    )
    S = librosa.logamplitude(np.abs(S), ref_power=np.max)
    
    times = librosa.frames_to_time(
        np.arange(len(S[0])), sr=sr, hop_length=hop_length, n_fft=nfft
    )
    freqs = librosa.fft_frequencies(sr=sr, n_fft=nfft)

    S_interp = scipy.interpolate.interp2d(
        times, freqs, S, kind='linear',
        copy=False, bounds_error=False,
        fill_value=0
    )
    return S_interp


def get_contours(trackid_list):
    contour_list = []
    instrument_list = []
    component_list = []
    contour_trackids = []
    for trackid in trackid_list:
        
        print(trackid)

        mix_path = os.path.join(MDB_MIXES, "{}_MIX.wav".format(trackid))
        mtrack = mdb.MultiTrack(trackid)
        
        for i, stem in mtrack.stems.items():
            data = stem.pitch_annotation

            if data is not None:
                print("    > {}".format(stem.instrument))

                conf = np.array(mtrack.activation_conf_from_stem(i)).T
                conf_interp = scipy.interpolate.interp1d(
                    conf[0], conf[1], fill_value=0, assume_sorted=True,
                    bounds_error=False
                )

                if trackid not in INTERPOLATORS.keys():
                    INTERPOLATORS[trackid] = salience_interpolator(mix_path)

                ctr = pitch_to_contours(
                    data, INTERPOLATORS[trackid], conf_interp, mix_path
                )
                contour_list.append(ctr)
                instrument_list.append(stem.instrument)
                component_list.append(stem.component)
                contour_trackids.append(trackid)

    return contour_list, instrument_list, component_list, contour_trackids


def get_vocal_label_dict(train_labels, test_labels, vocal_list):
    labels = list(set(train_labels + test_labels))
    label_dict = {lab: int(lab in vocal_list) for lab in labels}
    return label_dict


def get_melody_label_dict(train_labels, test_labels):
    labels = list(set(train_labels + test_labels))
    label_dict = {lab: int(lab == 'melody') for lab in labels}
    return label_dict


def get_bass_label_dict(train_labels, test_labels):
    labels = list(set(train_labels + test_labels))
    label_dict = {lab: int(lab == 'bass') for lab in labels}
    return label_dict


def get_inst_label_dict(train_labels, test_labels):
    labels = list(set(train_labels + test_labels))
    label_dict = {lab: i for i, lab in enumerate(labels)}
    return label_dict, labels


def build_training_testing_set(train_contours, train_labels, test_contours, test_labels,
                               label_dict):
    X_train = []
    Y_train = []
    for ctr, label in zip(train_contours, train_labels):
        label_idx = label_dict[label]
        feature = FTR.compute_all(ctr)
        X_train.append(feature)
        Y_train.append([label_idx]*len(ctr.nums))

    X_test = []
    Y_test = []
    for ctr, label in zip(test_contours, test_labels):
        label_idx = label_dict[label]
        feature = FTR.compute_all(ctr)
        X_test.append(feature)
        Y_test.append([label_idx]*len(ctr.nums))
    
    X_train = np.concatenate(X_train)
    X_test = np.concatenate(X_test)
    Y_train = np.concatenate(Y_train)
    Y_test = np.concatenate(Y_test)
    
    return X_train, Y_train, X_test, Y_test


def fit_model(X_train, Y_train, X_test, Y_test, class_labels, compute_prob=True):
    
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
    
    pprint(train_score)
    pprint(test_score)

    return train_score, test_score, clf


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
    outdir = args.results_path

    ## Aggregate tracks that will be used in experiment
    all_trackids = []
    all_trackids.extend(mdb.TRACK_LIST)
    # all_trackids.extend(mdb.TRACKLIST_V2)
    # all_trackids.extend(mdb.TRACKLIST_EXTRA)

    ## Create Train-test split
    data_split = mdb.utils.artist_conditional_split(
        trackid_list=all_trackids, test_size=0.2, num_splits=1, random_state=7
    )
    train_ids = data_split[0]['train']
    test_ids = data_split[0]['test']

    with open(os.path.join(outdir, 'train_test_split.json'), 'w') as fhandle:
        json.dump(data_split[0], fhandle, indent=2)

    ## Get ground truth contours for each stem
    print("Computing ground truth contours...")
    (train_contours, train_instrument_labels,
     train_component_labels, train_trackid) = get_contours(train_ids)
    (test_contours, test_instrument_labels,
     test_component_labels, test_trackid) = get_contours(test_ids)

    ## Run Vocal Non-Vocal Experiment
    print("Running Vocal experiment...")
    vocal_label_dict = get_vocal_label_dict(
        train_instrument_labels, test_instrument_labels, VOCALS
    )
    vocal_labels = ['non-vocal', 'vocal']
    X_train, Y_train, X_test, Y_test = build_training_testing_set(
        train_contours, train_instrument_labels,
        test_contours, test_instrument_labels,
        vocal_label_dict
    )
    train_score_vocal, test_score_vocal, vocal_clf = fit_model(
        X_train, Y_train, X_test, Y_test, vocal_labels
    )

    #TODO: Save stuff


    ## Run Melody Non-Melody Experiment
    print("Running Melody experiment...")
    melody_label_dict = get_melody_label_dict(
        train_component_labels, test_component_labels
    )
    melody_labels = ['non-melody', 'melody']
    X_train, Y_train, X_test, Y_test = build_training_testing_set(
        train_contours, train_component_labels,
        test_contours, test_component_labels,
        melody_label_dict
    )
    train_score_melody, test_score_melody, melody_clf = fit_model(
        X_train, Y_train, X_test, Y_test, melody_labels
    )

    #TODO: Save stuff

    ## Run Bass Non-Bass Experiment
    print("Running Bass experiment...")
    bass_label_dict = get_bass_label_dict(
        train_component_labels, test_component_labels
    )
    bass_labels = ['non-bass', 'bass']
    X_train, Y_train, X_test, Y_test = build_training_testing_set(
        train_contours, train_component_labels,
        test_contours, test_component_labels,
        bass_label_dict
    )
    train_score_bass, test_score_bass, bass_clf = fit_model(
        X_train, Y_train, X_test, Y_test, bass_labels
    )

    #TODO: Save stuff

    ## Run Instrument ID experiment
    print("Running Instrument ID experiment...")
    inst_label_dict, inst_labels = get_inst_label_dict(
        train_instrument_labels, test_instrument_labels
    )
    X_train, Y_train, X_test, Y_test = build_training_testing_set(
        train_contours, train_instrument_labels,
        test_contours, test_instrument_labels,
        inst_label_dict
    )
    train_score_inst, test_score_inst, inst_clf = fit_model(
        X_train, Y_train, X_test, Y_test, inst_labels, compute_prob=False
    )

    #TODO: Save stuff



if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Fill in missing time frames in a Tony file,"
                    "write the output to file.")
    PARSER.add_argument("results_path",
                        type=str,
                        help="Path to save results.")
    main(PARSER.parse_args())
