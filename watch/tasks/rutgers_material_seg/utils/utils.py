from glob import glob
from PIL import Image
import pickle as pkl
import os
import torch
import numpy as np
import yaml
import torch.nn.functional as F
import torchvision.transforms.functional as FT
import json
import random


def bandwise_norm(image, channel_first=True):

    if channel_first:
        bs, cs, h, w = image.shape
        for b in range(bs):
            for c in range(cs):
                single_channel = image[b, c, :, :]
                max_value = single_channel.max()
                min_value = single_channel.min()
                single_channel_normalized = (single_channel - min_value) / (max_value - min_value)
                image[b, c, :, :] = single_channel_normalized
    else:
        bs, h, w, cs = image.shape
        raise NotImplementedError

    return image


def otsu(image, num=400, get_bcm=False):
    c, h, w = image.shape
    image = image.view(1, -1)
    # print(image.shape)
    max_value = image.max()
    min_value = image.min()
    total_num = h * w
    step_value = (max_value - min_value) / num
    value = min_value + step_value
    best_inter_class_var = 0
    while value <= max_value:
        data_1 = image[image <= value]
        data_2 = image[image > value]
        # print(value)
        if data_1.shape[0] == 0 or data_2.shape[0] == 0:
            value += step_value
            continue
        w1 = data_1.shape[0] / total_num
        w2 = data_2.shape[0] / total_num

        mean_1 = data_1.mean()
        mean_2 = data_2.mean()

        inter_class_var = w1 * w2 * torch.pow((mean_1 - mean_2), 2)
        if best_inter_class_var < inter_class_var:
            best_inter_class_var = inter_class_var
            best_threshold = value
        value += step_value
    print(f"best: {best_threshold} max value: {max_value}")
    return best_threshold


def stad_image(image, channel_first=True, get_params=False, patches=False):
    if channel_first:
        if patches:
            bs, ps, c, h, w = image.shape
            image = image.reshape(bs, ps, c, h * w)  # (bs, ps, c, h*w)
            mean = image.mean(dim=3, keepdims=True)  # (bs, ps, c, 1)
            center = image - mean  # (bs, ps, c, h*w)
            var = torch.var(center, dim=3, keepdims=True)  # (bs, ps, c, h*w])
        else:
            bs, c, h, w = image.shape
            image = image.reshape(bs, c, h * w)  # (bs, c, h*w)
            mean = image.mean(dim=2, keepdims=True)  # (bs, c, 1)
            center = image - mean  # (bs, c, h*w)
            var = torch.var(center, dim=2, keepdims=True)  # (bs, c, h*w])
        # var = torch.sum(torch.pow(center,2), axis=2, keepdims=True) / (h * w) # (bs, c, 1)
        std = torch.sqrt(var)
        nm_image = center / std  # (bs, c, h*w)
        if patches:
            nm_image = nm_image.view(bs, ps, c, h, w)
        else:
            nm_image = nm_image.view(bs, c, h, w)

    if get_params:
        return nm_image, mean, std
    else:
        return nm_image


def denorm(image: torch.Tensor, mean: list = [0.485, 0.456, 0.406], std: list = [0.229, 0.224, 0.225]) -> torch.Tensor:
    """denorm shifted image

    Parameters
    ----------
    image : torch.Tensor
        image to denorm
    mean : list, optional
        mean applied to original image, by default [0.485, 0.456, 0.406]
    std : list, optional
        std applied to original image, by default [0.229, 0.224, 0.225]

    Returns
    -------
    torch.Tensor
        denromed image
    """
    if image.dim() == 3:
        assert image.dim() == 3, "Expected image [CxHxW]"
        assert image.size(0) == 3, "Expected RGB image [3xHxW]"

        for t, m, s in zip(image, mean, std):
            t.mul_(s).add_(m)
    elif image.dim() == 4:
        # batch mode
        assert image.size(1) == 3, "Expected RGB image [3xHxW]"

        for t, m, s in zip((0, 1, 2), mean, std):
            image[:, t, :, :].mul_(s).add_(m)

    return image


def create_dir_if_doesnt_exist(dir_path: str) -> None:
    """creates directory in given path

    Parameters
    ----------
    dir_path : str
        path to be created
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return


def normalize_image(image: np.array) -> np.array:
    """normalize image between 0-1

    Parameters
    ----------
    image : np.array

    Returns
    -------
    np.array
    """
    image_min = np.min(image)
    image_max = np.max(image)
    # print(f"image min: {image_min}, image_max: {image_max}")
    return (image - image_min) / (image_max - image_min + 0.00000000001)


def normalize_image_255(image: np.array) -> np.array:
    """normalizes image between 0-255

    Parameters
    ----------
    image : np.array
        image to be normalized

    Returns
    -------
    np.array
        normalized image
    """
    image_min = np.min(image)
    # image_max = np.max(image)
    # print(f"image min: {image_min}, image_max: {image_max}")
    return (image - image_min) / (255 - image_min + 0.00000000001)


def normalize_dm(image: np.array, confidence_score: float = 0) -> np.array:
    """normalize distance map

    Parameters
    ----------
    image : np.array
        distance map to be nromalized
    confidence_score : float, optional
        confidence score, by default 0

    Returns
    -------
    np.array
        normalized distance map
    """
    image_min = np.min(image)
    image_max = np.max(image)
    # print(f"image min: {image_min}, image_max: {image_max}")
    normalized = (image - image_min) / (image_max - image_min + 0.00000000001)
    normalized += confidence_score
    return normalized


def max_norm(p, version='torch', e=1e-5):
    if version == 'torch':
        if p.dim() == 3:
            C, H, W = p.size()
            p = F.relu(p)
            max_v = torch.max(p.view(C, -1), dim=-1)[0].view(C, 1, 1)
            min_v = torch.min(p.view(C, -1), dim=-1)[0].view(C, 1, 1)
            p = F.relu(p - min_v - e) / (max_v - min_v + e)
        elif p.dim() == 4:
            N, C, H, W = p.size()
            p = F.relu(p)
            max_v = torch.max(p.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
            min_v = torch.min(p.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
            p = F.relu(p - min_v - e) / (max_v - min_v + e)
    elif version in {'numpy', 'np'}:
        if p.ndim == 3:
            C, H, W = p.shape
            p[p < 0] = 0
            max_v = np.max(p, (1, 2), keepdims=True)
            min_v = np.min(p, (1, 2), keepdims=True)
            p[p < min_v + e] = 0
            p = (p - min_v - e) / (max_v + e)
        elif p.ndim == 4:
            N, C, H, W = p.shape
            p[p < 0] = 0
            max_v = np.max(p, (2, 3), keepdims=True)
            min_v = np.min(p, (2, 3), keepdims=True)
            p[p < min_v + e] = 0
            p = (p - min_v - e) / (max_v + e)
    return p


def batch_crf_inference(img: torch.Tensor, probs: torch.Tensor,
                        t: int = 1, scale_factor: int = 1, labels: int = 21) -> torch.Tensor:
    """crf inference for a batch

    Parameters
    ----------
    img : torch.Tensor
        image
    probs : torch.Tensor
        probablity map
    t : int, optional
    scale_factor : int, optional
    labels : int, optional
        number of labels, by default 21

    Returns
    -------
    torch.Tensor
        batch crf predictions
    """
    bs, c, h, w = probs.shape
    image_npy = img.numpy()
    probs_npy = probs.numpy()
    # preds = torch.zeros((bs, 1, h, w))
    preds_probs = torch.zeros((bs, labels, h, w))
    for b in range(bs):
        b_image = image_npy[b, :, :, :]
        b_probs = probs_npy[b, :, :, :]
        b_image = np.ascontiguousarray(np.transpose(b_image, (1, 2, 0)))
        b_image *= 255
        b_image[b_image > 255] = 255
        b_image[b_image < 0] = 0
        b_image = b_image.astype(np.uint8)
        b_pred = crf_inference(b_image, b_probs, t=t,
                               scale_factor=scale_factor, labels=labels)
        preds_probs[b, :, :, :] = torch.from_numpy(b_pred)
        # print(f"pred min: {b_pred.min()} max: {b_pred.max()}")
        # b_pred = b_pred.max(1)[1]
        # print(b_pred.shape)
        # preds[b,:,:,:] = torch.from_numpy(b_pred)
    # return preds, preds_probs
    return preds_probs


def crf_inference(img: np.array, probs: np.array, t: int = 10,
                  scale_factor: int = 1, labels: int = 21) -> np.array:
    """crf prediction for single image

    Parameters
    ----------
    img : np.array
        image
    probs : np.array
        probability map
    t : int, optional
    scale_factor : int, optional
    labels : int, optional

    Returns
    -------
    np.array
        single crf prediction
    """
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w = img.shape[:2]
    n_labels = labels
    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3 / scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80 / scale_factor, srgb=13,
                           rgbim=np.copy(img), compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))


def t2n(x: torch.Tensor) -> np.array:
    """convert tensor to numpy array

    Parameters
    ----------
    x : torch.Tensor
        tensor

    Returns
    -------
    np.array
        numpy array
    """
    x = x.cpu().detach().numpy()
    return x


def mat_to_csv(mat_path: str, save_to: str) -> None:
    """convert mat file to csv

    Parameters
    ----------
    mat_path : str
        mat file path
    save_to : str
        target path for csv file
    """
    import scipy.io
    import pandas as pd
    mat = scipy.io.loadmat(mat_path)
    mat = {k: v for k, v in mat.items() if k[0] != '_'}
    data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})
    data.to_csv(save_to)


def load_yaml_as_dict(yaml_path: str) -> dict:
    """load config file as dictionarry

    Parameters
    ----------
    yaml_path : str
        path to yaml file

    Returns
    -------
    dict
    """
    with open(yaml_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    return config_dict


def dictionary_contents(path: str, types: list,
                        recursive: bool = False) -> list:
    """extyract dictionary and subdictionary contents

    Parameters
    ----------
    path : str
        path of root
    types : list
        types of files to be extracted
    recursive : bool, optional
        search in subsequent directories, by default False

    Returns
    -------
    list
        list of files with full paths
    """
    files = []
    if recursive:
        path = path + "/**/*"
    for type in types:
        if recursive:
            for x in glob(path + type, recursive=True):
                files.append(os.path.join(path, x))
        else:
            for x in glob(path + type):
                files.append(os.path.join(path, x))
    return files


def save_pickle(object: object, path: str, file_name: str) -> None:
    """save pickle to location

    Parameters
    ----------
    object : object
        object to be saved
    path : str
        path to location
    file_name : str
        name of file to be saved
    """
    full_path = path + file_name + ".pkl"
    with open(full_path, 'wb') as file:
        pkl.dump(object, file)
    return


def load_pickle(path: str) -> object:
    """load pickle

    Parameters
    ----------
    path : str
        path to pickle file

    Returns
    -------
    object
        object loaded
    """
    with open(path, 'rb') as file:
        object = pkl.load(file)
    return object


def load_config_as_dict(path_to_config):
    with open(path_to_config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def config_parser(path_to_config, experiment_type):
    if experiment_type.lower() == "training":
        config = yaml.safe_load(open(path_to_config))
        return config

    elif experiment_type.lower() == "testing":
        raise Exception("incomplete parser for testing")


def load_json_as_dict(path_to_json):
    with open(path_to_json) as json_file:
        data = json.load(json_file)
    return data


def random_horizonal_flip(image: Image, mask: Image,
                          points: object = False) -> tuple:
    """random horizontal flip of both image and mask

    Parameters
    ----------
    image : PIL.Image
        image to be flipped
    mask : PIL.Image
        mask to be flipped
    points : bool, optional
        whether points are also provided. Can be either boolean or PIL.Image, by default False

    Returns
    -------
    tuple
        typle of image, mask, and possibly points
    """
    hflip = random.random() < 0.5
    if hflip:
        image = FT.hflip(image)
        mask = FT.hflip(mask)
        if isinstance(points, Image.Image):
            points = FT.hflip(points)
    if isinstance(points, Image.Image):
        return image, mask, points
    return image, mask


def random_vertical_flip(image: Image, mask: Image,
                         points: object = False) -> tuple:
    """random vertical flip of both image and mask

    Parameters
    ----------
    image : PIL.Image
        image to be flipped
    mask : PIL.Image
        mask to be flipped
    points : bool, optional
        whether points are also provided. Can be either boolean or PIL.Image, by default False

    Returns
    -------
    tuple
        typle of image, mask, and possibly points
    """
    vflip = random.random() < 0.5
    if vflip:
        image = FT.vflip(image)
        mask = FT.vflip(mask)
        if isinstance(points, Image.Image):
            points = FT.vflip(points)
    if isinstance(points, Image.Image):
        return image, mask, points
    return image, mask
