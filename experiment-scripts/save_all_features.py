"""Saves features for entire dataset"""
from __future__ import print_function

import numpy as np
import os
import utils
import argparse


def main(args):

    save_dir = args.results_path
    save_contour_path = args.save_contour_path
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(save_contour_path):
        os.mkdir(save_contour_path)

    ## Aggregate tracks that will be used in experiment
    all_trackids = utils.get_experiment_trackids()

    (all_contours, all_instrument_labels,
     all_component_labels, all_trackid, all_stemid) = utils.get_contours(
        all_trackids, save_contour_path, args.n_harms
    )

    print("getting features...")
    all_features = []
    contour_labels_instrument = []
    contour_labels_component = []
    contour_labels_trackid = []
    contour_labels_stemid = []
    for ctr, inst, component, trackid, stemid in zip(all_contours,
                                                     all_instrument_labels,
                                                     all_component_labels,
                                                     all_trackid,
                                                     all_stemid):
        feature = utils.FTR.compute_all(ctr)
        n_contours = len(ctr.nums)
        all_features.append(feature)
        contour_labels_instrument.append([inst]*n_contours)
        contour_labels_component.append([component]*n_contours)
        contour_labels_trackid.append([trackid]*n_contours)
        contour_labels_stemid.append([stemid]*n_contours)

    all_features = np.concatenate(all_features)
    contour_labels_instrument = np.concatenate(contour_labels_instrument)
    contour_labels_component = np.concatenate(contour_labels_component)
    contour_labels_trackid = np.concatenate(contour_labels_trackid)
    contour_labels_stemid = np.concatenate(contour_labels_stemid)

    with open(os.path.join(save_dir, 'features.npz'), 'w') as fhandle:
        np.savez(
            fhandle,
            features=all_features,
            instrument=contour_labels_instrument,
            component=contour_labels_component,
            trackid=contour_labels_trackid,
            stemid=contour_labels_stemid
        )


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Run contour classification experiments.")
    PARSER.add_argument("results_path",
                        type=str,
                        help="Path to save results.")
    PARSER.add_argument("save_contour_path",
                        type=str,
                        help="Path to save/find contours")
    PARSER.add_argument("n_harms",
                        type=int)
    main(PARSER.parse_args())
