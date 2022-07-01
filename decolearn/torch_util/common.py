import os
import numpy as np
import tifffile as tiff
import torch


def abs_helper(x, axis=1, is_normalization=True):
    x = torch.sqrt(torch.sum(x ** 2, dim=axis, keepdim=True))

    if is_normalization:
        for i in range(x.shape[0]):
            x[i] = (x[i] - torch.min(x[i])) / (torch.max(x[i]) - torch.min(x[i]) + 1e-16)

    x = x.to(torch.float32)

    return x


def check_and_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def dict2pformat(x: dict):
    ret = ''
    for k in x:
        ret += ' %s: [%.4f]' % (k, x[k])
    return ret


def dict2md_table(ipt: dict):
    ret = str()
    for section in ipt.keys():

        ret += '## ' + section + '\n'
        ret += '|  Key  |  Value |\n|:----:|:---:|\n'

        for i in ipt[section].keys():
            ret += '|' + i + '|' + str(ipt[section][i]) + '|\n'

        ret += '\n\n'

    return ret


def to_tiff(x, path, is_normalized=True):
    try:
        x = np.squeeze(x)
    except:
        pass

    try:
        x = torch.squeeze(x).numpy()
    except:
        pass

    print(x.shape, path)

    if len(x.shape) == 3:
        n_slice, n_x, n_y = x.shape

        if is_normalized:
            for i in range(n_slice):
                x[i] -= np.amin(x[i])
                x[i] /= np.amax(x[i])

                x[i] *= 255

            x = x.astype(np.uint8)

    else:
        n_slice, n_x, n_y, n_c = x.shape

        x = post_processing(np.expand_dims(x, -1))
        x = np.squeeze(x)

        if is_normalized:
            for i in range(n_slice):
                for j in range(n_c):
                    x[i, :, :, j] -= np.amin(x[i, :, :, j])
                    x[i, :, :, j] /= np.amax(x[i, :, :, j])

                    x[i, :, :, j] *= 255

            x = x.astype(np.uint8)

    tiff.imwrite(path, x, imagej=True, ijmetadata={'Slice': n_slice})


def write_test(log_dict, img_dict, save_path, is_save_mat=False, is_save_tiff=True):
    if log_dict:
        # Write Log_dict Information
        cvs_data = np.array(list(log_dict.values()))
        cvs_data = np.transpose(cvs_data, [1, 0])

        cvs_data_mean = cvs_data.mean(0)
        cvs_data_mean.shape = [1, -1]

        num_index = cvs_data.shape[0]
        cvs_index = np.arange(num_index) + 1
        cvs_index.shape = [-1, 1]

        cvs_data_with_index = np.concatenate([cvs_index, cvs_data], 1)

        cvs_header = ''
        for k in log_dict:
            cvs_header = cvs_header + k + ','

        np.savetxt(save_path + 'metrics.csv', cvs_data_with_index, delimiter=',', fmt='%.5f', header='index,' + cvs_header)
        np.savetxt(save_path + 'metrics_mean.csv', cvs_data_mean,  delimiter=',', fmt='%.5f', header=cvs_header)

        print(cvs_data_mean)

    if is_save_tiff:
        # Write recon of Img_Dict Information
        for key_ in ['fixed_y_tran', 'fixed_y_tran_recon', 'fixed_x',
                     'moved_y_tran', 'moved_y_tran_recon', 'moved_x',
                     'wrapped_f2m', 'wrapped_m2f']:

            if key_ in img_dict:
                print(key_, img_dict[key_].shape)
                to_tiff(img_dict[key_], save_path + key_ + '.tiff', is_normalized=False)