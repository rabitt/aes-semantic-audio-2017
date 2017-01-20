# CREATED: 1/20/17 17:18 PM by Justin Salamon <justin.salamon@nyu.edu>

import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import argparse
import os


VOCALS = ['male singer', 'female singer', 'male speaker', 'female speaker',
          'male rapper', 'female rapper', 'beatboxing', 'vocalists']


def contour_tsne(feature_file, feature_name_file, output_file, perplexity=50.0, n_iter=5000, min_duration=0):
    '''
    Compute TSNE from contour features and save to disk

    Parameters
    ----------
    feature_file : str
        Path to contour feature .npz file
    outout_file : str
        Path for saving TSNE output (npz)
    perplexity : float
        Perplexity parameter for TSNE
    n_iter : int
        Number of iterations for TSNE
    min_duration : float
        Minimum duration of contour (in seconds) to be included in TSNE

    Returns
    -------

    '''

    print('Loading features...')
    features = np.load(os.path.expanduser(feature_file))

    # Random state.
    random_state = 20170120

    # Get instrument list
    instruments = np.unique(features['instrument'])

    # Features
    X = np.vstack(features['features'])

    # Labels
    y_inst = np.hstack([instruments.tolist().index(inst) for inst in features['instrument']])
    y_mel = np.hstack([int(comp == 'melody') for comp in features['component']])
    y_voc = np.hstack([int(inst in VOCALS) for inst in features['instrument']])
    y_bas = np.hstack([int('bass' in inst) for inst in features['instrument']])

    # Scaler object for standardization
    scaler = StandardScaler()

    # Remove contours that are too short
    print('Filtering by duration...')
    names = np.load(os.path.expanduser(feature_name_file))
    duration_idx = names['feature_names'].tolist().index('duration')
    contour_idx = features['features'][:, duration_idx] >= min_duration
    Xsub = X[contour_idx]

    # Get corresponding labels
    ysub_inst = y_inst[contour_idx]
    ysub_mel = y_mel[contour_idx]
    ysub_voc = y_voc[contour_idx]
    ysub_bas = y_bas[contour_idx]

    # Standardize features
    print('Standardizing...')
    Xsub_std = scaler.fit_transform(Xsub)

    # Compute TSNE
    print('Computing TSNE projection...')
    projection = TSNE(random_state=random_state,
                      perplexity=perplexity,
                      n_iter=n_iter).fit_transform(Xsub_std)

    # Save projection, contour index and labels to disk
    results = {"tsne": projection,
               "contour_idx": contour_idx,
               "ysub_inst": ysub_inst,
               "ysub_mel": ysub_mel,
               "ysub_voc": ysub_voc,
               "ysub_bas": ysub_bas,
               "feature_file": feature_file,
               "feature_name_file": feature_name_file,
               "output_file": output_file,
               "perplexity": perplexity,
               "n_iter": n_iter,
               "min_duration": min_duration,
               "instruments": instruments,
               "random_state": random_state}

    print('Saving results to disk...')
    np.savez(os.path.expanduser(output_file), **results)

    print('Done.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('feature_file')
    parser.add_argument('feature_name_file')
    parser.add_argument('output_file')
    parser.add_argument('--perplexity', type=float, default=50.0)
    parser.add_argument('--n_iter', type=int, default=5000)
    parser.add_argument('--min_duration', type=float, default=0)

    args = parser.parse_args()

    contour_tsne(args.feature_file,
                 args.feature_name_file,
                 args.output_file,
                 args.perplexity,
                 args.n_iter,
                 args.min_duration)
