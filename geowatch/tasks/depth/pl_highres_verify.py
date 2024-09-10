# FIXME:
# Adds the "modules" subdirectory to the python path.
# See https://gitlab.kitware.com/smart/watch/-/merge_requests/148#note_1050127
# for discussion of how to refactor this in the future.
import geowatch_tpl  # NOQA

import warnings
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms

import pytorch_lightning as pl

from .backbone import get_backbone

from frame_field_learning import data_transforms
from frame_field_learning import local_utils
from frame_field_learning.model_multi import Multi_FrameFieldModel

from scipy import ndimage

dfactor = 25.5

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0


#-------------------------------------------
# Modify the batch_norm layers
#-------------------------------------------

def modify_bn(model, track_running_stats=True, bn_momentum=0.1):
    for m in model.modules():
        for child in m.children():
            if isinstance(child, nn.BatchNorm2d):

                child.momentum = bn_momentum
                child.track_running_stats = track_running_stats

                if track_running_stats is False:
                    child.running_mean = None
                    child.running_var =  None

    return model


#-------------------------------------------------
# Depth/Label/Shadow/Facade Eestimation Module
#-------------------------------------------------

class MultiTaskModel(pl.LightningModule):

    def __init__(
        self,
        batch_size: int = 1,
        checkpoint: str = None,
        config: dict = None,
        test_img_dir: str = None,
        test_img_list: str = None,
        gpus: str = '0',
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.gpus = gpus
        self.checkpoint = checkpoint
        self.batch_size = batch_size

        self.test_img_dir = test_img_dir
        self.test_img_list = test_img_list

        self.config = config

        self.backbone = get_backbone(self.config["backbone_params"])

        train_online_cuda_transform = None
        eval_online_cuda_transform = None

        self.net = Multi_FrameFieldModel(
            self.config,
            backbone=self.backbone,
            train_transform=train_online_cuda_transform,
            eval_transform=eval_online_cuda_transform)

        self.transform =  data_transforms.get_online_cuda_transform(
            self.config,
            augmentations=self.config["data_aug_params"]["enable"])

    def forward(self, x, tta=False):
        return self.net(x, tta)

    def test_step(self, batch, batch_idx):

        out_arr = []
        for i, image in enumerate(batch):
            if isinstance(image, dict):
                gid = image['id']
                # img_info = image
                image = image['imgdata']

            with torch.no_grad():

                image_float = image / 255.0
                mean = np.mean(image_float.reshape(-1, image_float.shape[-1]), axis=0)
                std = np.std(image_float.reshape(-1, image_float.shape[-1]), axis=0)

                batch2 = {
                    "image": torchvision.transforms.functional.to_tensor(image)[None, ...],
                    "image_mean": torch.from_numpy(mean)[None, ...],
                    "image_std": torch.from_numpy(std)[None, ...],
                }

                batch2 = local_utils.batch_to_cuda(batch2)

                pred2, batch2 = self(batch2, tta=True)

                output_depth = pred2['depth'][0, 0, :, :].cpu().data.numpy()
                output_label = pred2['seg'][0, 0, :, :].cpu().data.numpy()

                weighted_depth = dfactor * output_depth

                alpha = 0.9
                weighted_seg = alpha * output_label + (1.0 - alpha) * np.minimum(0.99, weighted_depth / 70.0)

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    tmp2 = 255 * anisotropic_diffusion(weighted_seg, niter=1, kappa=100, gamma=0.8)
                weighted_final = ndimage.median_filter(tmp2.astype(np.uint8), size=7)

                # Image.fromarray(weighted_final.astype(np.uint8)).save('/output/weighted_final.png')

            out_arr.append((gid, weighted_final))
        return out_arr

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = parent_parser.add_argument_group("MultiTaskModel")

        parser.add_argument('--checkpoint', default=None, type=str,
                            help='checkpoint to use for testing')
        parser.add_argument('--config', '--config', default=None, type=str,
                            help='Name of the config file, excluding the .json file extension.')
        parser.add_argument('--test_img_dir', '--test_img_dir', default=None, type=str,
                            help='directory where test images are located')
        parser.add_argument('--test_img_list', '--test_img_list', default=None, type=str,
                            help='list of test images')
        parser.add_argument('--gpus', default='0', type=str,
                            help='GPU')

        return parent_parser


# Vendored in to deal with 3.10 issue
def anisotropic_diffusion(img, niter=1, kappa=50,
                          gamma=0.1, voxelspacing=None, option=1):
    r"""
    Edge-preserving, XD Anisotropic diffusion.


    Parameters
    ----------
    img : array_like
        Input image (will be cast to numpy.float).
    niter : integer
        Number of iterations.
    kappa : integer
        Conduction coefficient, e.g. 20-100. ``kappa`` controls conduction
        as a function of the gradient. If ``kappa`` is low small intensity
        gradients are able to block conduction and hence diffusion across
        steep edges. A large value reduces the influence of intensity gradients
        on conduction.
    gamma : float
        Controls the speed of diffusion. Pick a value :math:`<= .25` for stability.
    voxelspacing : tuple of floats or array_like
        The distance between adjacent pixels in all img.ndim directions
    option : {1, 2, 3}
        Whether to use the Perona Malik diffusion equation No. 1 or No. 2,
        or Tukey's biweight function.
        Equation 1 favours high contrast edges over low contrast ones, while
        equation 2 favours wide regions over smaller ones. See [1]_ for details.
        Equation 3 preserves sharper boundaries than previous formulations and
        improves the automatic stopping of the diffusion. See [2]_ for details.

    Returns
    -------
    anisotropic_diffusion : ndarray
        Diffused image.

    Notes
    -----
    Original MATLAB code by Peter Kovesi,
    School of Computer Science & Software Engineering,
    The University of Western Australia,
    pk @ csse uwa edu au,
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal,
    Department of Pharmacology,
    University of Oxford,
    <alistair.muldal@pharm.ox.ac.uk>

    Adapted to arbitrary dimensionality and added to the MedPy library Oskar Maier,
    Institute for Medical Informatics,
    Universitaet Luebeck,
    <oskar.maier@googlemail.com>

    June 2000  original version. -
    March 2002 corrected diffusion eqn No 2. -
    July 2012 translated to Python -
    August 2013 incorporated into MedPy, arbitrary dimensionality -

    References
    ----------
    .. [1] P. Perona and J. Malik.
       Scale-space and edge detection using ansotropic diffusion.
       IEEE Transactions on Pattern Analysis and Machine Intelligence,
       12(7):629-639, July 1990.
    .. [2] M.J. Black, G. Sapiro, D. Marimont, D. Heeger
       Robust anisotropic diffusion.
       IEEE Transactions on Image Processing,
       7(3):421-432, March 1998.
    """
    # define conduction gradients functions
    import numpy
    if option == 1:
        def condgradient(delta, spacing):
            return numpy.exp(-(delta / kappa)**2.) / float(spacing)
    elif option == 2:
        def condgradient(delta, spacing):
            return 1. / (1. + (delta / kappa)**2.) / float(spacing)
    elif option == 3:
        kappa_s = kappa * (2**0.5)

        def condgradient(delta, spacing):
            top = 0.5 * ((1. - (delta / kappa_s)**2.)**2.) / float(spacing)
            return numpy.where(numpy.abs(delta) <= kappa_s, top, 0)

    # initialize output array
    out = numpy.array(img, dtype=numpy.float32, copy=True)

    # set default voxel spacing if not supplied
    if voxelspacing is None:
        voxelspacing = tuple([1.] * img.ndim)

    # initialize some internal variables
    deltas = [numpy.zeros_like(out) for _ in range(out.ndim)]

    for _ in range(niter):

        # calculate the diffs
        for i in range(out.ndim):
            slicer = tuple([slice(None, -1) if j == i else slice(None)
                           for j in range(out.ndim)])
            deltas[i][slicer] = numpy.diff(out, axis=i)

        # update matrices
        matrices = [
            condgradient(
                delta,
                spacing) *
            delta for delta,
            spacing in zip(
                deltas,
                voxelspacing)]

        # subtract a copy that has been shifted ('Up/North/West' in 3D case) by one
        # pixel. Don't as questions. just do it. trust me.
        for i in range(out.ndim):
            slicer = tuple([slice(1, None) if j == i else slice(None)
                           for j in range(out.ndim)])
            matrices[i][slicer] = numpy.diff(matrices[i], axis=i)

        # update the image
        out += gamma * (numpy.sum(matrices, axis=0))

    return out
