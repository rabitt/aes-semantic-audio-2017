from __future__ import print_function

import json
import librosa
import medleydb as mdb
from medleydb import download
import motif
from motif.core import Contours
import numpy as np
import os
import scipy
from sklearn.externals import joblib

INTERPOLATORS = {}
EXP_TRACKIDS = 'exp_trackids.txt'
SR = 44100.0/256.0
HOP = (1.0/SR) + 0.0001
F_THRESH = 0.8/12.0  # 80 cents
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
        return 0.025

    @classmethod
    def get_id(cls):
        return 'mock'

    def compute_contours(self, audio_filepath):
        pass


ETR = MockEtr()


def get_instrument(inst_list):
    mono_inst = [i for i in inst_list if mdb.multitrack.get_f0_type(i) == 'm']
    if len(mono_inst) > 0:
        return mono_inst[0]
    else:
        return inst_list[0]


def get_experiment_trackids():
    if os.path.exists(EXP_TRACKIDS):
        all_trackids = []
        with open(EXP_TRACKIDS, 'r') as fhandle:
            for line in fhandle.readlines():
                all_trackids.append(line.strip('\n'))
    else:
        mtracks = mdb.load_all_multitracks(
            dataset_version=['V1', 'V2', 'EXTRA', 'BACH10']
        )
        all_trackids = []
        for mtrack in mtracks:
            if mtrack.has_bleed:
                continue
            has_pitch_annot = [
                s.pitch_pyin_path is not None
                and os.path.exists(s.pitch_pyin_path)
                for s in mtrack.stems.values()
            ]
            if any(has_pitch_annot):
                print(mtrack.track_id)
                all_trackids.append(mtrack.track_id)
                download.download_mix(mtrack)

    return all_trackids


def save_contour(ctr, fpath):
    with open(fpath, 'w') as fhandle:
        np.savez(
            fhandle,
            index=ctr.index,
            times=ctr.times,
            freqs=ctr.freqs,
            salience=ctr.salience,
            sample_rate=ctr.sample_rate,
            audio_filepath=ctr.audio_filepath
        )


def load_contour(fpath):
    npzfile = np.load(fpath)
    index = npzfile['index']
    times = npzfile['times']
    freqs = npzfile['freqs']
    salience = npzfile['salience']
    sample_rate = int(npzfile['sample_rate'])
    audio_filepath = str(npzfile['audio_filepath'])
    ctr = Contours(
        index, times, freqs, salience, sample_rate, audio_filepath
    )
    return ctr


def save_to_json(dictionary, save_path):
    json_dict = jsonify_dictionary(dictionary)
    with open(save_path, 'w') as fhandle:
        json.dump(json_dict, fhandle, indent=2)


def save_classifier(motif_clf, save_path):
    joblib.dump(motif_clf.clf, save_path)


def load_classifier(filepath):
    motif_clf = motif.contour_classifiers.RandomForest()
    clf = joblib.load(filepath)
    motif_clf.clf = clf
    return motif_clf


def jsonify_dictionary(dictionary):
    jsonable_dictionary = {}
    for key, value in dictionary.items():
        if type(value) is np.array or type(value) is np.ndarray:
            jsonable_dictionary[key] = value.tolist()
        elif type(value) is dict:
            jsonable_dictionary[key] = jsonify_dictionary(value)
        else:
            jsonable_dictionary[key] = value

    return jsonable_dictionary


def pitch_to_contours(pitch_data, S_interp, conf_interp, mix_path,
                      n_harmonics=3):
    index = []
    times = []
    freqs = []
    salience = []

    i = 0
    index.append(i)
    times.append(pitch_data[0][0])
    freqs.append(pitch_data[0][1])
    salience.append(S_interp(pitch_data[0][0], pitch_data[0][1]))

    weights = 0.8**np.arange(n_harmonics)

    for (t_prev, f_prev), (t, f) in zip(pitch_data[:-1], pitch_data[1:]):

        if f == 0 or conf_interp(t) < 0.5:
            i = i + 1
            continue

        if ((t - t_prev > HOP) or f_prev == 0 or
                (np.abs(np.log2(f / f_prev)) > F_THRESH)):
            i = i + 1

        sal = np.average(
            S_interp(t, f * np.arange(n_harmonics)),
            weights=weights, axis=0
        )

        index.append(i)
        times.append(t)
        freqs.append(f)
        salience.append(sal)

    salience = np.array(salience)
    index, times, freqs, salience = ETR._postprocess_contours(
        np.array(index), np.array(times), np.array(freqs), salience.flatten()
    )

    #SMOOTH SALIENCE VALUES TODO: CHECK DIFFERENT KERNEL SIZES
    salience_smoothed = scipy.signal.medfilt(salience, kernel_size=5)
    contour = Contours(index, times, freqs, salience_smoothed, SR, mix_path)
    return contour


def pitch_to_contours_harmonics(pitch_data, S_interp, conf_interp, mix_path,
                                n_harmonics):
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

        # TODO: make motif support this!
        sal = S_interp(t, f*np.arange(n_harmonics))

        index.append(i)
        times.append(t)
        freqs.append(f)
        salience.append(sal)

    salience = np.array(salience)

    index, times, freqs, salience = ETR._postprocess_contours(
        np.array(index), np.array(times), np.array(freqs), salience.flatten()
    )

    #SMOOTH SALIENCE VALUES TODO: CHECK DIFFERENT KERNEL SIZES
    salience_smoothed = scipy.signal.medfilt(salience, kernel_size=5)
    contour = Contours(index, times, freqs, salience_smoothed, SR, mix_path)
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


def activation_conf_interpolator(mtrack, stem_idx):
    conf = np.array(mtrack.activation_conf_from_stem(stem_idx)).T
    conf_interp = scipy.interpolate.interp1d(
        conf[0], conf[1], fill_value=0, assume_sorted=True,
        bounds_error=False
    )
    return conf_interp


def get_vocal_label_dict(train_labels, test_labels, vocal_list):
    labels = list(set(train_labels + test_labels))
    label_dict = {lab: int(lab in vocal_list) for lab in labels}
    return label_dict


def get_function_label_dict(train_labels, test_labels):
    labels = sorted(list(set(train_labels + test_labels)))
    label_dict = {lab: i for i, lab in enumerate(labels)}
    return label_dict  


def get_melody_label_dict(train_labels, test_labels):
    labels = list(set(train_labels + test_labels))
    label_dict = {lab: int(lab == 'melody') for lab in labels}
    return label_dict


def get_bass_label_dict(train_labels, test_labels):
    labels = list(set(train_labels + test_labels))
    label_dict = {lab: int(lab == 'bass') for lab in labels}
    return label_dict


def get_vocaltype_label_dict(train_labels, test_labels, vocal_list):
    labels = list(set(train_labels + test_labels))
    label_dict = {lab: i for i, lab in enumerate(vocal_list)}
    for lab in list(set(train_labels + test_labels)):
        if lab not in label_dict:
            label_dict[lab] = -1
    return label_dict, labels


def get_inst_label_dict(train_labels, test_labels, inst_list):
    labels = list(set(train_labels + test_labels))
    label_dict = {lab: i for i, lab in enumerate(inst_list)}
    for lab in list(set(train_labels + test_labels)):
        if lab not in label_dict:
            label_dict[lab] = -1
    return label_dict, labels


def build_training_testing_set(train_contours, train_labels, test_contours,
                               test_labels, label_dict, delete_avg_pitch=False):
    X_train = []
    Y_train = []
    for ctr, label in zip(train_contours, train_labels):
        label_idx = label_dict[label]
        if label_idx == -1:
            continue
        feature = FTR.compute_all(ctr)
        X_train.append(feature)
        Y_train.append([label_idx]*len(ctr.nums))

    X_test = []
    Y_test = []
    for ctr, label in zip(test_contours, test_labels):
        label_idx = label_dict[label]
        if label_idx == -1:
            continue
        feature = FTR.compute_all(ctr)
        X_test.append(feature)
        Y_test.append([label_idx]*len(ctr.nums))

    X_train = np.concatenate(X_train)
    X_test = np.concatenate(X_test)
    Y_train = np.concatenate(Y_train)
    Y_test = np.concatenate(Y_test)

    if delete_avg_pitch:
        X_train = np.delete(X_train, 6, 1)
        X_test = np.delete(X_test, 6, 1)

    return X_train, Y_train, X_test, Y_test


def get_contours(trackid_list, save_contour_path, n_harms, use_pyin=True):
    contour_list = []
    instrument_list = []
    component_list = []
    contour_trackids = []
    contour_stemids = []

    for trackid in trackid_list:

        print(trackid)

        mtrack = mdb.MultiTrack(trackid)
        mix_path = mtrack.mix_path
        download.download_mix(mtrack)

        for i, stem in mtrack.stems.items():
            stem_identifier = os.path.basename(stem.audio_path).split('.')[0]
            contour_path = os.path.join(
                save_contour_path, "{}.npz".format(stem_identifier)
            )
            if os.path.exists(contour_path):
                has_contour = True
                data = None
            else:
                if use_pyin:
                    data = stem.pitch_estimate_pyin
                else:
                    data = stem.pitch_annotation
                has_contour = False

            if data is not None or has_contour:
                instrument = get_instrument(stem.instrument)
                print("    > {}".format(instrument))

                if has_contour:
                    print("      > loading contours...")
                    ctr = load_contour(contour_path)
                else:
                    print("      > computing contours...")
                    conf_interp = activation_conf_interpolator(
                        mtrack, i
                    )

                    if trackid not in INTERPOLATORS.keys():
                        INTERPOLATORS[trackid] = salience_interpolator(
                            mix_path
                        )

                    ctr = pitch_to_contours(
                        data, INTERPOLATORS[trackid], conf_interp, mix_path,
                        n_harmonics=n_harms
                    )

                    save_contour(ctr, contour_path)

                contour_list.append(ctr)
                instrument_list.append(instrument)
                component_list.append(stem.component)
                contour_trackids.append(trackid)
                contour_stemids.append(i)

    return (contour_list, instrument_list, component_list,
            contour_trackids, contour_stemids)

