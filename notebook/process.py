
import numpy as np



def contrast_normalization(arr_bin, tile_size = 128, extend=1.5):
    '''
    Computes the minimum and maximum contrast values to use when
    displaying a micrograph by calculating the mean of the medians 
    of the mic split up into tile_size * tile_size patches.

    :param arr_bin: the micrograph represented as a numpy array
    :type arr_bin: list
    :param tile_size: the size of the patch to split the mic by 
        (larger is faster)
    :type tile_size: int
    '''
    ny,nx = arr_bin.shape
    # set up start and end indexes to make looping code readable
    tile_start_x = np.arange(0, nx, tile_size)
    tile_end_x = tile_start_x + tile_size
    tile_start_y = np.arange(0, ny, tile_size)
    tile_end_y = tile_start_y + tile_size
    num_tile_x = len(tile_start_x)
    num_tile_y = len(tile_start_y)

    # initialize array that will hold means and std devs of all patches
    tile_all_data = np.empty((num_tile_y*num_tile_x, 2), dtype=np.float32)

    index = 0
    for y in range(num_tile_y):
        for x in range(num_tile_x):
            # cut out a patch of the mic
            arr_tile = arr_bin[tile_start_y[y]:tile_end_y[y], tile_start_x[x]:tile_end_x[x]]
            # store 2nd and 98th percentile values
            tile_all_data[index:,0] = np.percentile(arr_tile, 98)
            tile_all_data[index:,1] = np.percentile(arr_tile, 2)
            index += 1

    # calc median of non-NaN percentile values
    all_tiles_98_median = np.nanmedian(tile_all_data[:,0])
    all_tiles_2_median = np.nanmedian(tile_all_data[:,1])
    vmid = 0.5*(all_tiles_2_median+all_tiles_98_median)
    vrange = abs(all_tiles_2_median-all_tiles_98_median)
    # extend vmin and vmax enough to not include outliers
    vmin = vmid - extend*0.5*vrange
    vmax = vmid + extend*0.5*vrange

    return vmin, vmax

def preprocess(image):

    # image = Image.fromarray(image)
    # image = image.resize((image.width // BIN_FACTOR, image.height // BIN_FACTOR), Image.LANCZOS)
    # image = np.array(image)

    value_min, value_max = contrast_normalization(image)
    image = np.clip(image, value_min, value_max)
    mean, std = image.mean(), image.std()
    if std == 0:
        std = 1
    # image = (image-mean)/std
    return image, mean, std