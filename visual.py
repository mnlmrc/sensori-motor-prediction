import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


# from smp0.utils import detect_response_latency

# matplotlib.use('MacOSX')


def classicalMDS(G, contrast=None):
    """
    Classical MDS directly from the second moment matrix (adapted from Pcm for MATLAB).

    Parameters:
    G: numpy.ndarray
        KxK matrix form of the second moment matrix estimate.
    contrast: numpy.ndarray, optional
        A contrast matrix specifying the type of contrast optimized in the resultant representation.

    Returns:
    Y: numpy.ndarray
        Coordinates of the K conditions on the Q dimensions of the representational space.
    l: numpy.ndarray
        Eigenvalues for the importance of the different vectors.
    """

    # Making G symmetric
    G = (G + G.T) / 2

    # Eigen decomposition
    L, V = np.linalg.eig(G)
    l = np.sort(L)[::-1]  # Descending order
    V = V[:, np.argsort(L)[::-1]]

    # Scale by eigenvalues
    Y = V * np.sqrt(l)

    # If contrast is provided
    if contrast is not None:
        numVec = np.linalg.matrix_rank(contrast)
        H = contrast @ np.linalg.pinv(contrast)  # Projection matrix
        L = np.linalg.eigvalsh(Y.T.conj() @ H.T @ H @ Y)
        l = np.sort(np.real(L))[::-1]  # Descending order
        V = V[:, np.argsort(np.real(L))[::-1]]
        Y = Y @ V

    # Remove dimensions close to zero
    indx = np.where(l < np.finfo(float).eps)[0]
    Y[:, indx] = 0
    Y = np.real(Y)

    return Y, l


dict_text = {
    'xlabel': 'time (s)',
    'ylabel': None,
    'fs_label': 9,
    'xticklabels': None,
    'xticklabels_rotation': 45,
    'xticklabels_alignment': 'right',
    'fs_ticklabels': 7,
    'fs_title': 7,
    'fs_suptitle': 9,
}

dict_legend = {
    'fs': 6,
    'loc': 'upper center',
    'ncol': 5
}

dict_vlines = {
    'pos': list(),
    'ls': list(),
    'lw': list(),
    'color': list()
}

dict_lims = {
    'xlim': (None, None),
    'ylim': (0, None)
}

dict_bars = {
    'width': .2,
    'offset': .2
}

# def make_colors(n_labels, ecol=('blue', 'red')):
#     cmap = mcolors.LinearSegmentedColormap.from_list(f"{ecol[0]}_to_{ecol[1]}",
#                                                      [ecol[0], ecol[1]], N=100)
#     norm = plt.Normalize(0, n_labels)
#     colors = [cmap(norm(lab)) for lab in range(n_labels)]
#
#     return colors



