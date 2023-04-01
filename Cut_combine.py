import numpy as np


def cut(seismic_block, patch_size, stride_x, stride_y):

    [seismic_h, seismic_w] = seismic_block.shape

    n1 = 1
    while (patch_size + (n1 - 1) * stride_x) <= seismic_w:
        n1 = n1 + 1
    arr_w = patch_size + (n1 - 1) * stride_x
    n2 = 1
    while (patch_size + (n2 - 1) * stride_y) <= seismic_h:
        n2 = n2 + 1
    arr_h = patch_size + (n2 - 1) * stride_y
    fill_arr = np.zeros((arr_h, arr_w), dtype=np.float32)
    fill_arr[0:seismic_h, 0:seismic_w] = seismic_block

    path_w = []
    x = np.arange(n1)
    x = x + 1
    for i in x:
        s_x = patch_size + (i - 1) * stride_x
        path_w.append(s_x)
    number_w = len(path_w)
    path_h = []
    y = np.arange(n2)
    y = y + 1
    for k in y:
        s_y = patch_size + (k - 1) * stride_y
        path_h.append(s_y)
    number_h = len(path_h)
    cut_patches = []
    for index_x in path_h:
        for index_y in path_w:
            patch = fill_arr[index_x - patch_size: index_x, index_y - patch_size: index_y]
            cut_patches.append(patch)
    return cut_patches, number_h, number_w, arr_h, arr_w


def combine(patches, patch_size, number_h, number_w, block_h, block_w):

    temp = np.zeros((int(patch_size), 1), dtype=np.float32)
    for i in range(len(patches)):
        temp = np.append(temp, patches[i], axis=1)
    temp1 = np.delete(temp, 0, axis=1)

    test = np.zeros((1, int(patch_size*number_w)), dtype=np.float32)
    for j in range(0, int(patch_size*number_h*number_w), int(patch_size*number_w)):
        test = np.append(test, temp1[:, j:j + int(patch_size*number_w)], axis=0)
    test1 = np.delete(test, 0, axis=0)
    block_data = test1[0:block_h, 0:block_w]
    return block_data