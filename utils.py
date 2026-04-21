import csv

import astropy.io.fits as fits
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba import njit


#############
#   LAB 1   #
#############

def zeros(x, dtype=int):
    """
    custom zeros function to initialize empty arrays

    Usage:
        >> zeros(5, dtype=float)
        [0.0, 0.0, 0.0, 0.0, 0.0]

    :param x: numeric value, len of list to return
    :param dtype: dtype, type of values in list
    :return: list, empty array
    """
    return [dtype(0.0)] * int(x)


def my_ceil(x):
    """
    custom ceiling function

    Usage:
        >> my_ceil(4.5)
        5

    :param x: numeric value
    :return: float, ceiling of x
    """
    return -(-x // 1)


def ascend(ls, *others):
    """
    ascend a ls of numbers

    Usage:
        ls = [91, 82, 88]
        other1 = ["ben", "rick", "ashley"]
        other2 = ["grade 10", "grade 4", "grade 7"]
        ascend(ls, other1, other2)

    :param ls: list
    :param others: tuple, other lists to make same changes to (does not sort these)
    :return: None, modifies ls and others in-place
    """
    # set main pivot idx
    idx = 0
    while idx < len(ls):
        for i, x in enumerate(ls):
            # make sure we are past the main pivot
            if i > idx and x < ls[idx]:
                # swap pivot with current value
                ls[i], ls[idx] = ls[idx], x

                # loop through others and do the same
                for arr in others:
                    arr[i], arr[idx] = arr[idx], arr[i]

        # increment main pivot
        idx += 1

    return None


def ascend_str(ls, start_idx, end_idx):
    """
    sort a list of strs by specified numbers present in all strs

    Usage:
        ls =["ex03",
             "ex04",
             "ex01",
             "ex02",]
        ascend_str(ls, (-2, None))

    :param ls: list
    :param start_idx: int, start index of slice to sort
    :param end_idx: int, end index of slice to sort
    :return: None, modifies ls in-place
    """
    idx = 0
    # make slice for desired indices
    idx_slice = slice(start_idx, end_idx)
    while idx < len(ls):
        for i, x in enumerate(ls):
            # check for lowest integer at given indices
            if i > idx and int(x[idx_slice]) < int(ls[idx][idx_slice]):
                # swap pivot with current value
                ls[i], ls[idx] = ls[idx], x

        # increment main pivot
        idx += 1

    return None


def get_data(files):
    """
    grabs data from name of fits file

    Usage:
        files = ['data1.fits', 'data2.fits']
        data_array = get_data(files)

    :param files: str or list of strs, name of files to get data from
    :return: array, data array
    """
    # checks to see if files is a list, if not, make it one
    if type(files) is str:
        files = [files]
    n = len(files)

    # if only 1 item in list, just return the data for that str
    if n == 1:
        return fits.getdata(files[0])

    # otherwise initialize empty array and fill it with data
    data = zeros(n, dtype=float)
    for i, file in enumerate(files):
        data[i] = fits.getdata(file)

    return data


def get_exp(file):
    """
    get exposure time of a file

    Usage:
        exp = get_exp('data1.fits')

    :param file: str, name of file
    :return: int, exposure time of file
    """
    with fits.open(file) as hdu:
        hdr = hdu[0].header

    return hdr['EXPTIME']


def mean_variance(flat0, flat1):
    """
    calculate mean variance between two flats, (flat1-flat0)**2

    Usage:
        mean_var = mean_variance(flat0, flat1)

    :param flat0: array, Master frame, to be subtracted off from flat1
    :param flat1: array, Desired frame, from which flat0 is subtracted
    :return: float, avg variance of all pixels
    """
    # subtract master frame from desired frame
    diff = (flat1 - flat0).flatten()
    var = [d ** 2 for d in diff]
    return np.mean(var)


@njit
def hist_count(fdata, hmin, hmax):
    """
    Count function for histogram to be sped up by numba

    Usage:
        hist = hist_count(fdata, hmin, hmax)

    :param fdata: array, flattened data
    :param hmin: int, min x value
    :param hmax: int, max x value
    :return: array, filled frequency array
    """
    # initialize empty count arr (numba can only work with np.zeros but
    # I already wrote a custom zeros function so this should be fine)
    hist = np.zeros(int(np.ceil(hmax - hmin + 1)), dtype=np.int64)

    # count each value in bins
    for value in fdata:
        if hmin <= value <= hmax:
            hist[int(value - hmin)] += 1

    return hist


def histo(data, hmin=None, hmax=None, plot=True, xlabel=None, ylabel=None, title=None):
    """
    Plot histogram of data

    Usage:
        histo(data, 0, 2000, xlabel="x", ylabel="y", title="title")

    :param data: array, data
    :param hmin: int, min x value
    :param hmax: int, max x value
    :param plot: bool, whether to plot histogram or return arrays
    :param xlabel: str, label of x axis
    :param ylabel: str, label of y axis
    :param title: str, title of figure
    :return: None if plot else return a tuple of hr, hist arrays
    """
    if hmin is None:
        hmin = data.min()
    if hmax is None:
        hmax = data.max()

    fdata = data.flatten()

    # initialize bins
    hr = np.arange(hmin, hmax + 1)

    # count occurences
    hist = hist_count(fdata, hmin, hmax)

    if plot:
        plt.plot(hr, hist)
        plt.xlabel('ADU') if xlabel is None else plt.xlabel(f'{xlabel}')
        plt.ylabel('Frequency') if ylabel is None else plt.ylabel(f'{ylabel}')
        plt.title(title) if title is not None else None
        plt.show()
        return None
    else:
        return hr, hist


def set_negatives_to_zero_nd(tensor):
    """
    sets negative values to 0 in-place for a rank n tensor

    Usage:
        set_negatives_to_zero_nd(tensor)

    :param tensor: array, tensor
    :return: None, does in-place modifications to tensor
    """
    # check for rank 1
    ele = tensor[0]
    if isinstance(ele, np.ndarray):
        # not inside rank 1 yet so recursively loop with self call
        for sub in tensor:
            set_negatives_to_zero_nd(sub)
    else:
        # we are inside the rank 1 now
        for i, val in enumerate(tensor):
            if val < 0:  # if less than zero
                tensor[i] = 0  # set to zero

    return None


@njit
def poisson_log(x, mu):
    """
    Logarithmic approximation of Poisson distribution to avoid overflow error

    Usage:
        mu = mean(x_values)
        p_values = [poisson_log(x, mu) for x in x_values]

    :param x: float, x value, must be positive
    :param mu: float, mean of the distribution
    :return: float, approximate value of the Poisson distribution
    """
    if x == 0:
        log = -mu
    else:
        log = x * (np.log(mu) - np.log(x)) + x - mu
    return np.exp(log)


def get_poisson(data, xrange=None):
    """
    helper function to get poisson distribution arrays

    Usage:
        x, p = get_poisson(data, (0, 2000))

    :param data: array, data
    :param xrange: tuple, range of x values to be used in np.arange()
    :return: tuple of arrays, x_vals, p_vals
    """
    fdata = data.flatten()
    mu = np.mean(fdata)
    if xrange is None:
        x_min = int(fdata.min())
        x_max = int(fdata.max())
    else:
        x_min, x_max = xrange

    x_values = np.arange(x_min, x_max + 1)

    p_values = [poisson_log(x, mu) for x in x_values]

    return x_values, p_values


@njit
def gaussian(x, mu, sigma):
    """
    get value of gaussian distribution for given x, mu, sigma

    Usage:
        mu = mean(x_values)
        sigma = std(x_values)
        g_values = [gaussian(x, mu, sigma) for x in x_values]

    :param x: float, x value
    :param mu: float, avg of x distribution
    :param sigma: float, std of x distribution
    :return: float, value of gaussian distribution
    """
    coeff = 1 / (sigma * (2 * np.pi) ** 0.5)
    expo = np.exp(-(1 / 2) * ((x - mu) / sigma) ** 2)
    return coeff * expo


def get_gaussian(data, xrange=None):
    """
    helper function to get gaussian distribution arrays

    Usage:
        x, g = get_gaussian(data, (0, 2000))

    :param data: array, data
    :param xrange: tuple, range of x values to be used in np.arange()
    :return: tuple of arrays
    """
    fdata = data.flatten()
    mu = np.mean(fdata)
    sigma = np.std(fdata)
    if xrange is None:
        x_min = int(fdata.min())
        x_max = int(fdata.max())
    else:
        x_min, x_max = xrange

    x_values = np.arange(x_min, x_max + 1)

    g_values = [gaussian(x, mu, sigma) for x in x_values]

    return x_values, g_values


def test_colors(data, cmaps=None):
    """
    Test all pyplot cmaps (and reversed versions)  for given data

    Usage:
        cmaps = ['hot', 'afmhot']
        test_colors(data, cmaps)

    :param data: array, data
    :param cmaps: list of strs, specified names of cmaps, defaults to large list of very common cmaps
    :return: None
    """
    if cmaps is None:
        cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'gray', 'bone', 'pink',
                 'spring', 'summer', 'autumn', 'winter', 'cool', 'hot', 'afmhot', 'gist_heat', 'copper']

    # make copy of cmaps, otherwise we add to cmaps while iterating through it, infinitely
    cmaps_copy = cmaps.copy()
    for i, cmap in enumerate(cmaps_copy):
        # as we edit cmaps, the indices change, 2*i+1 keeps up with these changes
        cmaps.insert(2 * i + 1, cmap + '_r')

    for cmap in cmaps:
        plt.imshow(data, cmap=cmap)
        plt.title(cmap)
        plt.show()

    return None


def combine_legend(fig, **legend_kwargs):
    """
    combines legend elements of all axes in a fig

    Usage:
        combine_legend(fig)

    :param fig: plt figure
    :param legend_kwargs: kwargs for legend
    :return: None, legend will be placed
    """
    # get axes
    axes = fig.get_axes()

    # init empty lists
    combined_handles = []
    combined_labels = []

    # loop through axes
    for ax in axes:
        # grab legend info
        handles, labels = ax.get_legend_handles_labels()
        # add to lists (extend() because function above could return a list if multiple items on certain axis)
        combined_handles.extend(handles)
        combined_labels.extend(labels)

    # perform on last axis, for display reasons
    axes[-1].legend(combined_handles, combined_labels, **legend_kwargs)

    return None


#############
#   LAB 2   #
#############


def parse_csv_between_markers(file_path, start_marker, end_marker):
    """
    parse data in csv between two sentinal values, start_marker, end_marker

    Usage:
        data = parse_csv_between_markers(file_path, start_marker, end_marker)

    :param file_path: str, name of file
    :param start_marker: str, start marker
    :param end_marker: str, end marker
    :return: list of data from csv
    """
    data_lines = []
    recording = False

    # open with read mode and set newline characters to empty string
    with open(file_path, 'r', newline='') as f:
        for line in f:
            stripped_line = line.strip()  # remove excess whitespace
            if stripped_line == end_marker:
                break  # stop recording when end_marker is found

            elif recording:
                data_lines.append(line)

            elif stripped_line == start_marker:
                recording = True  # start recording after start_marker is found

    reader = csv.reader(data_lines)
    return list(reader)


def get_centroids(wavelengths, intensities, threshold=None, threshold_lim=0.01, scope=20, return_indices=True):
    """
    gets centroid wavelengths for all peaks in intensity above certain threshold
    for a peak to count it has to be largest in scope radius
    meaning largest out of [scope] number of points forward and back

    also returns variance of centroids

    Usage:
        get_centroids(wavelengths, intensities, threshold=0.01, scope=20)
        get_centroids(wavelengths, intensities, threshold_lim=0.005, scope=5)

    :param wavelengths: array-like of wavelengths
    :param intensities: array-like of intensities
    :param threshold: float, threshold value, adjust in tandem with scope
        if None, automatically determine threshold to be 1.1x mean value of intensities
    :param threshold_lim: float, minimum threshold if automatically determining threshold, default 0.01
    :param scope: int, scope radius, default 20,
        larger scope ensures no false peaks within small bump regions
        smaller scope allows for more peaks to be found (potentially false peaks too)
    :param return_indices: bool, whether or not to return indices of all_centroids (pixel numbers)
    :return: if return_indices, (peaks, centroids, error), otherwise (centroids, error)
    """
    if threshold is None:
        threshold = 1.1 * np.mean(intensities)
        if threshold < threshold_lim:
            threshold = threshold_lim

    n = len(intensities)
    peak_indices = []
    for i, I in enumerate(intensities):
        # dont count end points ("i-n" is negative version of index)
        if i < scope or (i - n) >= -scope:
            continue

        if I > threshold:
            # build like [i+scope, i+(scope-1), ..., i-(scope-1), i-scope]
            vals = [intensities[i - j] for j in range(-scope, scope + 1) if j != 0]

            # make sure intensity is peak of scope
            if I > max(vals):
                peak_indices.append(i)

    centroids = []
    centroid_errors = []
    for k, peak_idx in enumerate(peak_indices.copy()):
        # go left until intensity drops below threshold
        left = peak_idx
        while left > 0 and intensities[left - 1] > threshold:
            left -= 1

        # repeat right
        right = peak_idx
        while right < n - 1 and intensities[right + 1] > threshold:
            right += 1

        regional_wavelengths = wavelengths[left:right + 1]
        regional_intensities = intensities[left:right + 1]
        # matmul acts as sum of element-wise product with 1d arrays
        centroid = np.matmul(regional_wavelengths, regional_intensities) / my_sum(regional_intensities)

        # Prevent repeats and keep same length
        if centroid in centroids:
            peak_indices.remove(peak_idx)
        else:
            centroids.append(centroid)
            # error prop for centroid, var = sigma^2
            variance = np.matmul(regional_intensities,
                                 (regional_wavelengths - centroid) ** 2) / my_sum(regional_intensities) ** 2
            centroid_errors.append(variance)

    if return_indices:
        return np.array(peak_indices), np.array(centroids), np.array(centroid_errors)
    else:
        return np.array(centroids), np.array(centroid_errors)


def linear_least_squares(x, y, weights=None):
    """
    Linear Least Square Fit

    :param x: np.array
    :param y: np.array
    :param weights: np.array, optional error associated with x
    :return: m, c tuple representing (slope, intercept)
    """
    # ensure numpy
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    # set weights if provided
    w = 1. / (np.asarray(weights, dtype=np.float32) ** 2) if weights is not None else np.ones_like(x)

    # construct design matrix B: each row is [x_i, 1]
    B = np.column_stack((x, np.ones_like(x)))

    # compute weighted normal matrix
    W = np.diag(w)
    BTW = B.T @ W
    Cov = np.linalg.inv(BTW @ B)  # 2x2 covariance matrix

    # finalize
    params = Cov @ (B.T @ (w * y))
    m, c = params[0], params[1]
    return m, c, Cov


def frame_sub(x, frame, single=False):
    """
    subtracts frame from x

    :param x: first frame
    :param frame: frame to be subtracted from x
    :param single: bool, whether or not x is a singular frame
    :return: reduced frame
    """
    if single:
        return x - frame

    n = len(x)
    tot = zeros(n, dtype=float)

    for i, data in enumerate(x):
        tot[i] = data - frame
    return tot


def normalize_flats(data):
    """
    normalizes flats by reducing by median

    :param data: 3d-array, all frames to be reduced
    :return: normalized flat
    """
    n = len(data)
    tot = zeros(n, dtype=float)

    for i, d in enumerate(data):
        tot[i] = d / (np.median(d))

    return np.median(tot, axis=0)


def norm(data, norm_flat):
    """
    normalizes frame by reducing by a normalized flat

    :param data: 2d-array, frame to be normalized
    :param norm_flat: normalized flat frame
    :return: normalized frame
    """
    return data / (norm_flat + 1e-2)  # prevent divide by zero


#############
#   LAB 3   #
#############


def remove_bad_cols(x, bad_cols):
    """
    removes bad columns from x by setting them to background

    :param x: 2d array, data with bad columns
    :param bad_cols: int or list, index or indices of bad columns
    :return: 2d array, frame with bad columns set to background
    """
    x[:, bad_cols] = np.median(x)
    return x


def get_hdr_data(file, entry):
    """
    get header value for a file

    Usage:
        exp = get_hdr_data('data1.fits', 'EXPTIME')

    :param file: str, name of file
    :param entry: str, name of header entry storing desired value
    :return: object, value of header entry
    """
    with fits.open(file) as hdu:
        hdr = hdu[0].header

    try:
        return hdr[entry]
    except KeyError:
        print(f"Could not find {entry} in {file}")
        return None


def load_headers_all_files(headers, data_files=None, data_dir=None):
    """
    load values for each header entry for all files in a 2D array
    rows being headers, and cols being data_file

    :param headers: list, list of header entries
    :param data_files: list, list of data files
    :param data_dir: str, optional dir_prefix for all data files
    :return: np.array, 2d array of values for each header for each file
    """
    if data_dir is None or not isinstance(data_dir, str):
        data_dir = globals().get('data_dir')
        if data_dir is None:
            data_dir = ""

    if data_files is None:
        data_files = globals().get('data_files')
        if data_files is None:
            raise ValueError("No available 'data_files' has been defined.")

    # All loaded headers will be saved here. np.full() creates a NumPy array of a given shape (first argument)
    # filled with a constant value (second argument, empty string in this case). "dtype = object" will allow
    # the array to store data of any type (some headers may be numbers, not strings).
    output = np.full([len(headers), len(data_files)], "", dtype=object)

    # Now read the headers from all files

    # YOUR CODE HERE
    for i, hdr in enumerate(headers):
        for j, file in enumerate(data_files):
            output[i, j] = get_hdr_data(data_dir + file, hdr)

    return output


def load_frame_add_pedestal(filename):
    """
    load frame and add pedestal from FITS header

    :param filename: str, name of file
    :return: data frame with added pedestal
    """
    frame = get_data(filename)
    pedestal = get_hdr_data(filename, 'PEDESTAL')
    if pedestal is not None:
        # if neg, change to positive then subtract (can't add a negative to uint)
        if pedestal < 0:
            frame = frame - np.full_like(frame, -pedestal)
            set_negatives_to_zero_nd(frame)
        else:
            frame += pedestal

    return frame


def load_frame_subtract_bias(filename, bias, bad_col_idx=None):
    """
    load frame and subtract the frame and remove overscan

    :param filename: str or list, name or list of names of data_file
    :param bias: int or array, int or frame to subtract
    :param bad_col_idx: int, index of bad column
    :return: 2d or 3d np.array, clean frame or frames
    """
    if isinstance(filename, str):
        frame = load_frame_add_pedestal(filename)
        if isinstance(bias, int):
            bias = np.full_like(frame, bias)
        image = frame - bias
        set_negatives_to_zero_nd(image)

        # set bad col to background noise
        if isinstance(bad_col_idx, int):
            image[:, bad_col_idx] = np.median(image)

        return image

    elif isinstance(filename, list):
        images = zeros(len(filename), np.float32)
        for i, fname in enumerate(filename):
            frame = load_frame_add_pedestal(fname)
            if isinstance(bias, int):
                bias = np.full_like(frame, bias)
            images[i] = frame - bias
            set_negatives_to_zero_nd(images[i])

            # set bad col to background noise
            if isinstance(bad_col_idx, int):
                images[i][:, bad_col_idx] = np.median(images[i])

        return images
    else:
        return None


def load_reduced_science_frame(filename, flat, bias):
    """
    load reduced science frame

    :param filename: str or list, path to frame file
    :param flat: array, normalized, cleaned flat frame
    :param bias: array, frame frame
    :return: array, normalized reduced frame(s)
    """
    # frame sub data
    data_clean = load_frame_subtract_bias(filename, bias)

    # norm with clean flat
    norm = data_clean / (flat + 1e-2)

    return norm


def master_frame(x, reduction='median'):
    """
    creates master frame by reducing frames, adds built-in pedestal to frame if available

    :param x: list of frame path strings
    :param reduction: str, 'mean' or 'median', default 'median'
    :return: master frame array
    """
    n = len(x)
    tot = zeros(n, dtype=float)

    for i, file in enumerate(x):
        arr = load_frame_add_pedestal(file)
        tot[i] = arr

    if reduction == 'mean':
        return np.mean(tot, axis=0)
    elif reduction == 'median':
        return np.median(tot, axis=0)
    else:
        raise ValueError("reduction must be either 'mean' or 'median'")


# TODO: write this
def write_master_fits():
    pass


def plot_im(ax, data, xlabel='x (px)', ylabel='y (px)', title='', fname=None, **imshow_kwargs):
    """
    Plot an image using matplotlib imshow.

    :param ax: plt.Axes, axes to plot on
    :param data: array, data to imshow
    :param xlabel: str, x label
    :param ylabel: str, y label
    :param title: str, title
    :param fname: str, filename
    :param imshow_kwargs: iterable, imshow kwargs including colorbar axis kwargs 'pad' and 'size'
    :return: None, plots data using imshow
    """
    # extract cax kwargs from imshow_kwargs
    cax_keys = ['pad', 'size']
    cax_kwargs = {key: imshow_kwargs.pop(key) for key in cax_keys if key in imshow_kwargs}
    cax_kwargs.setdefault('size', "5%")
    cax_kwargs.setdefault('pad', 0.05)

    fig = ax.get_figure()
    ax.imshow(data, **imshow_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=cax_kwargs['size'], pad=cax_kwargs['pad'])
    fig.colorbar(ax.images[0], cax=cax)
    if fname is None:
        return None
    else:
        fig.savefig(fname)
        return None


def find_star_locs(im_data, n_size=10, bright_count_thresh=10, background_factor=2):
    """
    ARGUMENTS:
    ================================================================
    im_data - Reduced 2D FITS Data
    n_size  - Neighborhood size to consider for windowing (pixels)
    bright_count_thresh - Threshold for number of 'bright'pixels in
                          neighborhood to be considered a star.
                          (Proportional to size of Star blob)
    background_factor - factor to multiply background by to set
                        definition of bright pixel
    RETURNS:
    ================================================================
    [[x_positions_of_star_center, y_positions_of_star_center]]
    i.e., a list of list of x and y coordinate of star centers
    """

    # set definition of a 'bright' pixel to be 3 times the background
    background = np.median(im_data)
    count_threshold = background_factor * background

    star_centers = []

    # indexing uses x=rows, y=cols
    # but for analysis we want x=cols and y=rows, so we must swap when referencing
    ny, nx = im_data.shape

    for y in range(0, ny, n_size):
        for x in range(0, nx, n_size):
            # check to see if we're in either left corner (bad corners), if so, skip
            if (y > 979 or y < 50) and x < 50:
                continue

            # set window
            window = im_data[y:y + n_size, x:x + n_size]

            count_bright = np.sum(window > count_threshold)
            if count_bright >= bright_count_thresh:
                center_y = y + window.shape[0] // 2
                center_x = x + window.shape[1] // 2

                star_centers.append([center_y, center_x])

    return star_centers


def make_pos_array(image):
    """
    makes image position array with same shape as image

    :param image: 2D array, science frame
    :return: position array, same shape as image
    """
    row_dim = np.shape(image)[0]
    row_pos = np.array(range(row_dim))[np.newaxis]
    pos_xarr = np.tile(row_pos.T, (1, row_dim))
    pos_yarr = np.tile(row_pos, (row_dim, 1))
    pos_arr = np.dstack((pos_xarr, pos_yarr))

    return pos_arr


def collapse_subsets(arr_list):
    """
    remove all subsets from a list, only keep the super sets (sets that are not subsets of another set)

    :param arr_list: list or array-likes
    :return: list of super sets present in arr_list
    """
    # convert arrays to sets
    sets = [set(arr) for arr in arr_list]
    keep = [True] * len(arr_list)

    # check for subsets
    for i, s in enumerate(sets):
        for j, t in enumerate(sets):
            if i != j:
                # if s is a strict (not duplicate) subset of t, mark it to be removed.
                if s.issubset(t) and len(s) < len(t):
                    keep[i] = False
                    break

    # remove duplicates
    filtered = []
    for flag, arr in zip(keep, arr_list):
        if flag and arr not in filtered:
            filtered.append(arr)

    return filtered


def calc_centroids_2d(intarr, posarr, loc_list, window_max=20):
    """
    given intensities, positions, locations, and a window size, calculate the
    centroid position value for each window at the specified location

    PARAMETERS:
    ==================================================================================
    intarr - array of intensities (image), shape [y, x]
    posarr - array of positions, shape [y, x, 2]
    loc_list - list of [y, x] (NOT [x, y]!) coords to calculate centroids around
    window_max - Size of Window to consider to find max pos of each star (in pixels)

    RETURNS:
    ==================================================================================
    centroids - List of list of centroid coordinates and corresponding uncertainities
                Format: [[xc, yc, unc_xc, unc_yc]]
    """

    centroids = []

    window_size = window_max // 2

    for i, (y, x) in enumerate(loc_list):
        # check edges
        if x < window_size or y < window_size or y > np.shape(intarr)[0] - window_size or x > np.shape(intarr)[1] - window_size:
            # centroids.append([float('NaN')]*4)
            continue

        # window off region
        y_slice = slice(y - window_size, y + window_size)
        x_slice = slice(x - window_size, x + window_size)
        region_ints = intarr[y_slice, x_slice]
        region_pos = posarr[y_slice, x_slice, :]

        # denominator
        tot_int = np.sum(region_ints)

        # matrix version of equation above
        centroid = np.einsum('ijk,ij->k', region_pos, region_ints) / tot_int

        # error prop
        diff = region_pos - centroid

        # matrix version of eq above (diff transposed is represented by the different indices in einsum)
        cov = np.einsum('ij,ijk,ijl->kl', region_ints, diff, diff) / tot_int ** 2
        sig_y, sig_x = cov[0, 0] ** .5, cov[1, 1] ** .5

        # round pixels to whole numbers
        centroid_full = [round(centroid[0]), round(centroid[1]), sig_y, sig_x]

        centroids.append(centroid_full)

    centroids = np.array(centroids, dtype='object')

    # remove marks for same cluster
    # set threshold window to half window_max in pixels
    threshold = window_max // 2
    all_neighbor_indices = []
    for i, (y, x, sy, sx) in enumerate(centroids):
        # check threshold and make mask for those that cross it
        diff = np.abs(centroids - centroids[i])
        mask = (diff[:, 0] <= threshold) & (diff[:, 1] <= threshold)
        neighbor_indices = np.where(mask)[0]

        # only store clusters, not single star locs
        if len(neighbor_indices) > 1:
            all_neighbor_indices.append(list(neighbor_indices))

    # remove subsets to only get the main clusters
    clusters = collapse_subsets(all_neighbor_indices)
    keep = [True] * len(centroids)
    for cluster in clusters:
        # collapse along vertical dimension to average all values
        collapsed_cluster = np.mean(centroids[cluster], axis=0)

        # overwrite first occurence of cluster with the averaged version of its neighbors
        centroids[cluster[0]] = collapsed_cluster

        # mark neighbors for removal (everything after first occurence of cluster)
        for idx in cluster[1:]:
            keep[idx] = False

    centroids = centroids[keep]

    return centroids


def local_pixel_size(ra_deg, dec_deg, center_coord, focal_length=16480, pixel_size=0.030, offset=512, standard=False):
    """
    Converts RA and DEC from degrees to x pixel and y pixel using plate constants

    Params:
    ra_deg: Right Ascension of the object in degrees
    dec_deg: Declination of the object in degrees
    center_coord: AstroPy SkyCoord object, coordinate of center of image
    standard: bool, if True, returns standard coordinates only, no local conversion
    focal_length: focal length in mm
    pixel_size: pixel size in mm
    offset: pixel offset in pixels

    Returns: the x pixel and y pixel locations of the object
    """
    # convert all values from deg to radians
    ra = ra_deg * np.pi / 180
    dec = dec_deg * np.pi / 180
    ra_0 = center_coord.ra.value * np.pi / 180
    dec_0 = center_coord.dec.value * np.pi / 180

    # calculate common denominator beforehand
    denom = (np.cos(dec_0)*np.cos(dec)*np.cos(ra-ra_0) + np.sin(dec) * np.sin(dec_0))

    # standard coordinates from ra and dec
    X = - np.cos(dec) * np.sin(ra-ra_0)/denom
    Y = - (np.sin(dec_0)*np.cos(dec)*np.cos(ra-ra_0) - np.cos(dec_0)*np.sin(dec))/denom

    if standard:
        return X, Y

    # convert to pixels using plate constants and center using offset
    x = focal_length*X/pixel_size + offset
    y = focal_length*Y/pixel_size + offset

    return x, y


def local_plate_scale(ra_deg, dec_deg, center_coord, plate_scale=0.368, offset=512):
    """
    Converts RA and DEC from degrees to x pixel and y pixel using plate scale

    Params:
    ra_deg: Right Ascension of the object in degrees
    dec_deg: Declination of the object in degrees
    center_coord: AstroPy SkyCoord object, coordinate of center of image
    plate_scale: plate scale in as/px
    offset: pixel offset in pixels

    Returns: the x pixel and y pixel locations of the object
    """
    # get relative to center coord and change from deg to arcsec
    ra = (ra_deg - center_coord.ra.value) * 3600
    dec = (dec_deg - center_coord.dec.value) * 3600

    # convert from as to px using plate scale
    x = ra / plate_scale
    y = dec / plate_scale

    # centering by offset
    x += offset
    y += offset

    return x, y


def sky_query(dataframe, filename, fov_width='6.3m', fov_height='6.3m', magnitude_limit=18):
    """
    vizier query the sky for objects around center of a file

    :param dataframe: pandas.DataFrame, dataframe containing the data of file, must have columns
                            ['FILE NAME'] ['RA'] ['DEC'] ['DATE-BEG'] ['RADECSYS']
    :param filename: str, name of file in dataframe
    :param fov_width: str, width of field of view in arcs, '6.3m' is 6.3 arcmin
    :param fov_height: str, height of field of view in arcs, '6.3m' is 6.3 arcmin
    :param magnitude_limit: float, R2 magnitude limit
    :return: ra, dec arrays of queried objects
    """
    # grab necessary values from provided dataframe
    ra_center, dec_center, yr = dataframe.loc[dataframe['FILE NAME'] == filename, ['RA', 'DEC', 'DATE-BEG']].values[0]
    reference_frame = dataframe.loc[dataframe['FILE NAME'] == filename, 'RADECSYS'].values[0].lower()

    center_coord = SkyCoord(ra=ra_center, dec=dec_center, unit=(u.hour, u.deg), frame=reference_frame)

    vizier = Vizier(column_filters={"R2mag": f"<{magnitude_limit}"})
    result_table = vizier.query_region(center_coord, width=fov_width, height=fov_height, catalog="USNO-B1")

    # Extract required data from obtained query results
    ra_cat = np.array(result_table[0]["RAJ2000"])  # this is the stars' RA in the 2000 epoch
    dec_cat = np.array(result_table[0]["DEJ2000"])  # this is the stars' Dec in the 2000 epoch
    pm_ra = np.array(result_table[0]["pmRA"])  # this is the RA proper motion of the stars
    pm_dec = np.array(result_table[0]["pmDE"])  # this is the Dec proper motion of the stars
    mag = np.array(result_table[0]["R2mag"])

    # convert mas/yr to deg/yr
    pm_ra = pm_ra / 1000 / 3600
    pm_dec = pm_dec / 1000 / 3600

    # time in years since epoch (2000)
    dt = yr - 2000

    # add proper motion to epoch coordinates
    ra_cat = ra_cat + pm_ra * dt
    dec_cat = dec_cat + pm_dec * dt

    return ra_cat, dec_cat, center_coord


def nearest_neighbor_match(a, b):
    """
    Matches entries by euclidean distance

    Parameters:
        a : ndarray of shape (N,2)
            Array containing x and y positions [x, y].
        b : ndarray of shape (M,2)
            Array containing x and y positions [x, y].

    Returns:
        matches : list of tuples
            Each tuple is (a_index, b_index) indicating a match.
    """
    # Ensure inputs are numpy arrays
    a = np.array(a).astype(np.float32)
    b = np.array(b).astype(np.float32)

    N = a.shape[0]
    matches = []
    used_centroids = set()

    for i in range(N):
        a_i = a[i]  # This should be a (2,) array

        # euclidean distance:
        diff = b - a_i  # Shape (M,2)
        distances = np.sqrt(np.sum(diff**2, axis=1))  # Shape (M,)

        sorted_indices = np.argsort(distances)
        for j in sorted_indices:
            if j not in used_centroids:
                matches.append((i, j))
                used_centroids.add(j)
                break

    return matches


def get_1_sigma_region(x, cov):
    """
    Gets 1 sigma region from covariance matrix

    :param x: np.ndarray, independent variable
    :param cov: np.ndarray, covariance matrix
    :return: np.ndarray, 1-sigma region
    """
    # ensure numpy

    sigma_m = cov[0, 0] ** .5
    sigma_c = cov[1, 1] ** .5
    cov_mc = cov[0, 1]
    sigma_y = (x ** 2 * sigma_m ** 2 + sigma_c ** 2 + 2 * x * cov_mc) ** .5
    return sigma_y




