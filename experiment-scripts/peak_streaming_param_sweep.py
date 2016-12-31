"""Script to run Justin's peak-streaming parameter sweep.
"""
from __future__ import print_function

import argparse
import csv
import medleydb
import mir_eval
import motif
import numpy as np
import random
import time

TRACK_IDS = [
    'MusicDelta_Country1', 'MusicDelta_Beatles', 'ClaraBerryAndWooldog_Boys',
    'MusicDelta_Rockabilly', 'Schubert_Erstarrung', 'AmarLal_SpringDay1'
]
HOP_LENGTH = 128
WIN_LENGTH = 2048
NFFT = 8192
HRANGE = [1, 2, 3, 4, 5]
HWEIGHTS = np.power(0.8, np.array(HRANGE) - 1)

def load_annot_multif0(fpath):
    "load an annotation into multi-f0 format"
    times = []
    freqs = []
    with open(fpath, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter=',')
        for line in reader:
            times.append(float(line[0]))
            f = np.array(line[1:], dtype=float)
            f = f[f > 0]
            freqs.append(f)
    return np.array(times), freqs


def get_random_parameterization():
    "get a contour extractor with a random parameter setting"
    pitch_cont = random.uniform(30, 100)
    max_gap = random.choice(np.arange(127, 2206, 128)/44100.0)
    amp_thresh = random.uniform(0, 1)
    dev_thresh = random.uniform(0, 4)
    
    etr = motif.contour_extractors.peak_stream.PeakStream(
        hop_length=HOP_LENGTH, win_length=WIN_LENGTH, n_fft=NFFT,
        h_range=HRANGE, h_weights=HWEIGHTS, interpolation_type='linear',
        pitch_cont=pitch_cont, max_gap=max_gap, amp_thresh=amp_thresh,
        dev_thresh=dev_thresh, preprocess=True, use_salamon_salience=True
    )

    params = [pitch_cont, max_gap, amp_thresh, dev_thresh]

    return etr, params


def preload_salience(mtracks):
    salience_data = {}
    etr, _ = get_random_parameterization()
    for mtrack in mtracks:
        print(mtrack.track_id)

        salience_data[mtrack.track_id] = {}
        if etr.preprocess:
            fpath = etr._preprocess_audio(
                mtrack.mix_path, normalize_format=True,
                normalize_volume=True
            )
        else:
            fpath = mtrack.mix_path

        times, freqs, S = etr._compute_salience_salamon(fpath)
        salience_data[mtrack.track_id]['times'] = times
        salience_data[mtrack.track_id]['freqs'] = freqs
        salience_data[mtrack.track_id]['S'] = S

    return salience_data

def get_contours(etr, times, freqs, S, mix_path):
    psh = etr.PeakStreamHelper(
        S, times, freqs, etr.amp_thresh, etr.dev_thresh, etr.n_gap,
        etr.pitch_cont
    )

    c_numbers, c_times, c_freqs, c_sal = psh.peak_streaming()
    if len(c_numbers) > 0:
        c_numbers, c_times, c_freqs, c_sal = etr._sort_contours(
            np.array(c_numbers), np.array(c_times), np.array(c_freqs),
            np.array(c_sal)
        )
        (c_numbers, c_times, c_freqs, c_sal) = etr._postprocess_contours(
            c_numbers, c_times, c_freqs, c_sal
        )

    return Contours(
        c_numbers, c_times, c_freqs, c_sal, etr.sample_rate,
        mix_path
    )


def main(args):
    mtracks = list(medleydb.load_multitracks(TRACK_IDS))

    salience_data = preload_salience(mtracks)

    row_data = []

    for i in range(args.n_iter):
        print("running iteration {}".format(i))
        
        etr, params = get_random_parameterization()
        print("pitch_cont: {}, max_gap: {}".format(params[0], params[1]))
        print("amp_thresh: {}, dev_thresh: {}".format(params[2], params[3]))

        try:
            recall = []
            precision = []
            accuracy = []
            for mtrack in mtracks:

                print("    > {}".format(mtrack.track_id))
                start_time = time.time()

                annot_fpath = mtrack.melody3_fpath

                print("        * computing contours...")
                ctr = get_contours(
                    etr,
                    salience_data[mtrack.track_id]['times'],
                    salience_data[mtrack.track_id]['freqs'],
                    salience_data[mtrack.track_id]['S'],
                    mtrack.mix_path
                )

                print("        * to multif0 format...")
                est_time, est_freqs = ctr.to_multif0_format()
                ref_time, ref_freqs = load_annot_multif0(annot_fpath)

                print("        * scoring...")
                scores = mir_eval.multipitch.evaluate(
                    np.array(ref_time), ref_freqs,
                    np.array(est_time), est_freqs
                )

                recall.append(scores['Recall'])
                precision.append(scores['Precision'])
                accuracy.append(scores['Accuracy'])

                elapsed_time = time.time() - start_time
                print("      elapsed time: {}".format(elapsed_time))

            row = [
                np.mean(accuracy), np.std(accuracy),
                np.mean(precision), np.std(precision),
                np.mean(recall), np.std(recall),
                params[0], params[1], params[2], params[3]
            ]

            print("")
            row_data.append(row)

            with open(args.output_path, "wb") as f:
                writer = csv.writer(f)
                writer.writerows(row_data)
        except:
            print("There was a problem...moving on!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fill in missing time frames in a Tony file,"
                    "write the output to file.")
    parser.add_argument("results_path",
                        type=str,
                        help="Path to save results.")
    parser.add_argument("n_iter",
                        type=int,
                        help="Number of iterations to run.")
    main(parser.parse_args())

