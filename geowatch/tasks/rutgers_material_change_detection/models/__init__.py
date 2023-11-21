import os

import torch
import torch.nn as nn
import torchvision.models as tv_models

import geowatch.tasks.rutgers_material_change_detection.models.resnet as resnet
from geowatch.tasks.rutgers_material_change_detection.utils.util_misc import load_cfg_file
from geowatch.tasks.rutgers_material_change_detection.models.timesformer import TimeSformer
from geowatch.tasks.rutgers_material_change_detection.models.dynamic_unet import DynamicUNet
from geowatch.tasks.rutgers_material_change_detection.models.mat_ed_framework import MatED, MTMatED
from geowatch.tasks.rutgers_material_change_detection.models.simple_cnn_encoder import SimpleCNNEncoder
from geowatch.tasks.rutgers_material_change_detection.models.peri_resnet import resnet34 as peri_resnet34
from geowatch.tasks.rutgers_material_change_detection.models.simple_decoder import SimpleDecoder, Decoder
from geowatch.tasks.rutgers_material_change_detection.models.late_fusion_framework import LateFusionFramework
from geowatch.tasks.rutgers_material_change_detection.models.early_fusion_framework import EarlyFusionFramework
from geowatch.tasks.rutgers_material_change_detection.models.discritizers import Gumbel_Softmax, VectorQuantizer2
from geowatch.tasks.rutgers_material_change_detection.models.early_fusion_mat_framework import EarlyFusionMatFramework
from geowatch.tasks.rutgers_material_change_detection.models.patch_transformer_framework import PatchTransformerFramework

from geowatch.tasks.rutgers_material_change_detection.models.patch_transformer import (
    PatchTransformerEncoder,
    PatchTransformerDecoder,
)
from geowatch.tasks.rutgers_material_change_detection.models.self_attention import (
    SelfAttention,
    AsymmetricPyramidSelfAttention,
)


def build_model(cfg, video_slice, n_in_channels, max_frames, n_out_channels, device=None):
    task_mode = cfg.task_mode
    framework = cfg.framework

    # Get pretrain information.
    pretrain = cfg.framework.pretrain

    if framework.name in ["early_fusion", "patch_transformer", "attn_space_time", "late_fusion"]:

        # Get model encoder.
        ## Hamdle special early fusion cases.
        if framework.name == "early_fusion":
            n_in_channels = max_frames * n_in_channels

        # print(cfg.weights_load)
        if cfg.weights_load == "all":
            pretrain_path = pretrain
        if cfg.weights_load != "encoder":
            pretrain = None

        encoder = get_encoder(cfg, cfg.framework.encoder_name, n_in_channels, max_frames, pretrain)

        # Compute encoder output sizes.
        feat_sizes = compute_encoder_output_sizes(cfg.framework.name, encoder, n_in_channels, video_slice, max_frames)

        # Get sequence model.
        sequence_model = get_sequence_model(cfg.framework.sequence_name, cfg, max_frames, feat_sizes, device)

        # Get model decoder.
        decoder = get_decoder(cfg, cfg.framework.decoder_name, feat_sizes, n_in_channels, n_out_channels, task_mode)

        # Get attention modules.
        attention_modules = get_attention(cfg.framework.name, feat_sizes, cfg)

        # Get model framework.
        if framework.name == "early_fusion":
            framework = EarlyFusionFramework(task_mode, encoder, decoder, attention_modules)
        elif framework.name == "patch_transformer":
            framework = PatchTransformerFramework(
                task_mode, encoder, decoder, None, first_last_feat_aux_loss=cfg.framework.first_last_feat_aux_loss
            )
        elif framework.name == "late_fusion":
            framework = LateFusionFramework(
                task_mode,
                encoder,
                decoder,
                attention_modules,
                sequence_model,
                cfg=cfg,
                feat_sizes=feat_sizes,
                device=device,
            )

        if cfg.weights_load == "all":
            print(f"loading weights from {pretrain_path}")
            framework = load_pretrained_weights(framework, pretrain=pretrain_path, freeze_encoder=cfg.framework.freeze)
    elif framework.name in ["mat_ed", "mt_mat_ed"]:
        # Build encoder.
        encoder = get_encoder(cfg, cfg.framework.encoder_name, n_in_channels, max_frames, pretrain)

        # Compute encoder output sizes.
        feat_sizes = compute_encoder_output_sizes(cfg.framework.name, encoder, n_in_channels, video_slice, max_frames)

        # Build discretizer.
        output_index = int(cfg.framework.feat_layer[-1]) - 1
        if cfg.framework.discretizer == "gumbel_softmax":
            discretizer = Gumbel_Softmax(
                cfg.framework.n_classes, feat_sizes[output_index][0], out_feat_dim=cfg.framework.codeword_reduce_dim
            )
        elif cfg.framework.discretizer is None:
            discretizer = None
        elif cfg.framework.discretizer == "vec_quantizer":
            discretizer = VectorQuantizer2(
                cfg.framework.n_classes,
                feat_sizes[output_index][0],
                out_feat_dim=cfg.framework.codeword_reduce_dim,
                beta=cfg.framework.vq_beta,
            )
        else:
            raise NotImplementedError(f'Discretizer named "{cfg.framework.discretizer}" not implmented.')

        if cfg.framework.codeword_reduce_dim is None:
            codeword_feats_dim = feat_sizes[0][0]
        else:
            codeword_feats_dim = cfg.framework.codeword_reduce_dim

        # Get attention block
        if cfg.framework.attention_name == "self_attention":
            attention_block = SelfAttention(cfg.framework.name, feat_sizes, [True], cfg.framework.attention_n_heads)
        elif cfg.framework.attention_name is None:
            attention_block = None
        else:
            raise NotImplementedError(
                f'Attention type "{cfg.framework.attention_name}" not implemented for mat_ed framework.'
            )

        try:
            decoder_sa = cfg.framework.self_attention
        except AttributeError:
            decoder_sa = True

        # Build decoder.
        decoder = Decoder(
            ch=codeword_feats_dim,
            out_ch=n_in_channels,
            num_res_blocks=1,
            attn_resolutions=[1],
            z_channels=codeword_feats_dim,
            resolution=feat_sizes[0][1],
            in_channels=codeword_feats_dim,
            ch_mult=[1, 2],
            self_attention=decoder_sa,
        )

        if framework.name == "mat_ed":
            framework = MatED(cfg.task_mode, encoder, discretizer, decoder, attention_block=attention_block)
        elif framework.name == "mt_mat_ed":
            framework = MTMatED(cfg.task_mode, encoder, discretizer, decoder, attention_block=attention_block)
    elif framework.name == "early_fusion_mat":
        # Get model encoder.
        if cfg.framework.mat_integration == "features":
            ef_n_in_channels = max_frames * (n_in_channels + cfg.framework.mat_feat_proj_dim)
        elif cfg.framework.mat_integration == "change_conf":
            ef_n_in_channels = (max_frames + 1) * n_in_channels
        else:
            raise NotImplementedError

        encoder = get_encoder(cfg, cfg.framework.encoder_name, ef_n_in_channels, max_frames, pretrain)

        # Compute encoder output sizes.
        feat_sizes = compute_encoder_output_sizes(
            cfg.framework.name, encoder, ef_n_in_channels, video_slice, max_frames
        )

        # Get sequence model.
        sequence_model = get_sequence_model(cfg.framework.sequence_name, cfg, max_frames, feat_sizes, device)

        # Get model decoder.
        decoder = get_decoder(cfg, cfg.framework.decoder_name, feat_sizes, ef_n_in_channels, n_out_channels, task_mode)

        # Get attention modules.
        attention_modules = get_attention(cfg.framework.name, feat_sizes, cfg)

        # TODO: Build material encoder model.

        ## Load mat_config file.
        mat_config_path = os.path.join(cfg.framework.mat_model_dir, "config.yaml")
        mat_config = load_cfg_file(mat_config_path)

        ## Build encoder.
        mat_encoder = get_encoder(mat_config, mat_config.framework.encoder_name, n_in_channels, max_frames, pretrain)

        # Compute encoder output sizes.
        feat_sizes = compute_encoder_output_sizes(
            mat_config.framework.name, mat_encoder, n_in_channels, video_slice, max_frames
        )

        ## Build discretizer.
        output_index = int(mat_config.framework.feat_layer[-1]) - 1
        if mat_config.framework.discretizer == "gumbel_softmax":
            discretizer = Gumbel_Softmax(
                mat_config.framework.n_classes,
                feat_sizes[output_index][0],
                out_feat_dim=mat_config.framework.codeword_reduce_dim,
            )
        elif mat_config.framework.discretizer is None:
            discretizer = None
        elif mat_config.framework.discretizer == "vec_quantizer":
            discretizer = VectorQuantizer2(
                mat_config.framework.n_classes,
                feat_sizes[output_index][0],
                out_feat_dim=mat_config.framework.codeword_reduce_dim,
                beta=mat_config.framework.vq_beta,
            )
        else:
            raise NotImplementedError(f'Discretizer named "{mat_config.framework.discretizer}" not implmented.')

        if mat_config.framework.codeword_reduce_dim is None:
            codeword_feats_dim = feat_sizes[0][0]
        else:
            codeword_feats_dim = mat_config.framework.codeword_reduce_dim

        ## Get attention block
        if mat_config.framework.attention_name == "self_attention":
            attention_block = SelfAttention(
                mat_config.framework.name, feat_sizes, [True], mat_config.framework.attention_n_heads
            )
        elif mat_config.framework.attention_name is None:
            attention_block = None
        else:
            raise NotImplementedError(
                f'Attention type "{mat_config.framework.attention_name}" not implemented for mat_ed framework.'
            )

        mat_decoder = None

        if mat_config.framework.name == "mat_ed":
            mat_framework = MatED(
                mat_config.task_mode, mat_encoder, discretizer, mat_decoder, attention_block=attention_block
            )
        elif mat_config.framework.name == "mt_mat_ed":
            mat_framework = MTMatED(
                mat_config.task_mode, mat_encoder, discretizer, mat_decoder, attention_block=attention_block
            )

        # TODO: Load pretrained weights of material encoder model.
        mat_model_data_path = os.path.join(cfg.framework.mat_model_dir, "best_model.pth.tar")
        mat_model_data = torch.load(mat_model_data_path)
        mat_framework.load_state_dict(mat_model_data["state_dict"], strict=False)

        # TODO: Set parameters to not requiring gradient.
        for param in mat_framework.encoder.parameters():
            param.requires_grad = False
        for param in mat_framework.discretizer.parameters():
            param.requires_grad = False
        if attention_block is not None:
            for param in mat_framework.attention_block.parameters():
                param.requires_grad = False

        # Build final framework.
        framework = EarlyFusionMatFramework(
            cfg.task_mode,
            encoder,
            decoder,
            attention_modules,
            mat_framework,
            cfg.framework.mat_feat_proj_dim,
            cfg.framework.mat_integration,
        )

    else:
        print(f'Framework "{cfg.framework.name}" not implemented')
        exit()

    # Get pretrained weights.
    if pretrain is not None:
        framework = load_pretrained_weights(framework, pretrain, freeze_encoder=cfg.framework.freeze)

    return framework


def get_encoder(cfg, encoder_name, n_channels, max_frames, pretrain=None):
    """Contruct an encoder to encode image(s) into features.

    Args:
        cfg (?): TODO
        encoder_name (str): Name of the encoder.
        n_channels (int): Number of channels that go into the encoder.
        max_frames (int): Maximum number of frames in a video.
        pretrain (str, optional): [description]. Defaults to None.
        freeze (bool, optional): [description]. Defaults to False.

    Raises:
        NotImplementedError: [description]

    Returns:
        encoder (nn.Module): [description]
        feat_dim (list[int]): [description]
    """
    if encoder_name[:6] == "resnet":
        encoder = load_resnet_model(encoder_name, n_channels, pretrain)
    elif encoder_name == "simple_cnn_unet":
        encoder = SimpleCNNEncoder(n_channels, bilinear=True)
    elif encoder_name == "patch_transformer":
        assert cfg.height == cfg.width, "Input frame must be a square."
        input_shape = cfg.height * cfg.scale
        patch_shape = cfg.framework.patch_size
        encoder = PatchTransformerEncoder(
            cfg.task_mode,
            n_channels,
            input_shape,
            patch_shape,
            max_frames,
            dim=cfg.framework.input_dim,
            n_heads=cfg.framework.n_heads,
            n_blocks=cfg.framework.n_blocks,
            dim_linear_block=cfg.framework.dim_linear_block,
            p_dropout=cfg.framework.p_dropout,
        )
    elif encoder_name == "timesformer":
        encoder = TimeSformer(
            cfg.height,
            patch_size=cfg.framework.patch_size,
            num_frames=max_frames,
            embed_dim=cfg.framework.encoder.proj_dim,
            n_blocks=cfg.framework.encoder.n_blocks,
            n_heads=cfg.framework.encoder.n_heads,
            mlp_ratio=cfg.framework.encoder.mlp_ratio,
            n_in_channels=n_channels,
        )
    else:
        raise NotImplementedError(f'Encoder name "{encoder_name}" not implemented for get_encoder function.')

    return encoder


def load_resnet_model(encoder_name, n_channels, pretrain=None, freeze=False):
    # Gets randomly initialized weights of model.
    if encoder_name == "resnet18":
        model = resnet.resnet18()
    elif encoder_name == "resnet34":
        model = resnet.resnet34()
    elif encoder_name == "resnet50":
        model = resnet.resnet50()
    elif encoder_name == "resnet_peri_34":
        model = peri_resnet34()

    # Get pretrained weights.
    if pretrain is not None:
        # Get pretrained model weights.
        if pretrain == "imagenet":
            print("Loading ImageNet weights.")
            if encoder_name == "resnet18":
                pt_model_weights = tv_models.resnet18(True)
            elif encoder_name == "resnet34":
                pt_model_weights = tv_models.resnet34(True)
            elif encoder_name == "resnet50":
                pt_model_weights = tv_models.resnet50(True)
            else:
                raise NotImplementedError(f"Have not implemented {encoder_name} pretrained weights.")

            # Load pretrained weights into backbone model.
            model.load_state_dict(pt_model_weights.state_dict())
        else:
            if encoder_name == "resnet_peri_34":
                print("INFO: Loading weights from Peri's model.")
                model = peri_resnet34(pretrain)

    # Update first convolution layer.
    if n_channels != 3:
        # Update the number of input channels.
        model.conv1 = nn.Conv2d(
            n_channels,
            model.conv1.out_channels,
            kernel_size=model.conv1.kernel_size,
            stride=model.conv1.stride,
            padding=model.conv1.padding,
            bias=model.conv1.bias,
        )

        # Initialize with Xavier weights.
        nn.init.xavier_uniform_(model.conv1.weight)

    return model


def get_decoder(cfg, decoder_name, feat_sizes, n_in_channels, n_out_channels, task_mode):
    if task_mode in ["ss_arrow_of_time", "ss_splice_change", "ss_splice_change_index"]:
        decoder = SimpleDecoder(feat_sizes, n_out_channels)
        return decoder

    if decoder_name == "unet":
        decoder = DynamicUNet(feat_sizes, n_in_channels, n_out_channels)
    elif decoder_name == "patch_transformer":
        decoder = PatchTransformerDecoder(
            feat_sizes, task_mode, n_out_channels, cfg.framework.decoder_agg_mode, cfg.framework.patch_size
        )
    elif decoder_name == "transformer":
        # Just use regular encoder.

        decoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=cfg.framework.encoder.proj_dim,
            nhead=cfg.framework.decoder.n_heads,
            dim_feedforward=cfg.framework.decoder.feat_dim,
            dropout=0.0,
        )
        decoder = torch.nn.TransformerEncoder(decoder_layer, cfg.framework.decoder.n_blocks)
    else:
        raise NotImplementedError(f"No implementation for decoder of name: {decoder_name}")

    return decoder


def compute_encoder_output_sizes(framework_name, encoder, n_in_channels, video_slice, max_frames):
    if framework_name in ["early_fusion", "late_fusion", "mat_ed", "mt_mat_ed", "early_fusion_mat"]:

        # Compute size of the input tensor.
        fake_input = torch.zeros(
            [1, n_in_channels, video_slice.height * video_slice.scale, video_slice.width * video_slice.scale]
        )
        with torch.no_grad():
            output = encoder(fake_input)

        feat_sizes = [list(out.shape)[1:] for out in output.values()]

    elif framework_name == "patch_transformer":
        feat_sizes = [max_frames, encoder.dim]

    elif framework_name == "attn_space_time":
        feat_sizes = []

    else:
        raise NotImplementedError(f'Get feature size for framework "{framework_name}" not implemented.')

    return feat_sizes


def get_sequence_model(seq_model_name, cfg, max_frames, feat_sizes, device=None):
    if seq_model_name in ["max_feats", "mean_feats"]:
        sequence_model = seq_model_name
    elif seq_model_name == "patch_transformer":
        if device is None:
            device = "cuda"
        # Params.
        n_heads = 6
        n_blocks = 4
        dim_linear_block = 512
        p_dropout = 0.0

        active_layers = [False, False, False, True]

        sequence_model = {}
        for i, (feat_size, active_layer) in enumerate(zip(feat_sizes, active_layers)):
            # Compute input shape.
            n_channels = feat_size[0]
            input_shape = feat_size[1]
            assert feat_size[1] == feat_size[2]

            # Compute patch shape
            patch_shape = 6
            while (input_shape % patch_shape) != 0:
                patch_shape -= 1
                if patch_shape == 1:
                    break

            # Create patch transformer.
            if active_layer:
                sequence_model["layer" + str(i + 1)] = PatchTransformerEncoder(
                    cfg.task_mode,
                    n_channels,
                    input_shape,
                    patch_shape,
                    max_frames,
                    dim=n_channels * patch_shape**2,
                    n_heads=n_heads,
                    n_blocks=n_blocks,
                    dim_linear_block=dim_linear_block,
                    p_dropout=p_dropout,
                ).to(device)
            else:
                sequence_model["layer" + str(i + 1)] = "max_feats"
    elif seq_model_name is None:
        # Special case for basic material reconstruction self-supervised task.
        sequence_model = seq_model_name
    else:
        raise NotImplementedError(f'No get sequence model call for "{seq_model_name}".')

    return sequence_model


def load_pretrained_weights(framework, pretrain, freeze_encoder=False):
    if pretrain is None:
        return framework

    # Load all model weights
    # Load weights from saved model.
    if os.path.isdir(pretrain):
        # Load weights from best model.
        weight_path = os.path.join(pretrain, "best_model.pth.tar")
        model_data = torch.load(weight_path)

        framework.load_state_dict(model_data["state_dict"], strict=False)

    # Check if model weights should be frozen.
    if freeze_encoder:
        print("Freezing encoder weights.")
        for param in framework.encoder.parameters():
            param.requires_grad = False

    return framework


def get_attention(framework_name, feat_sizes, cfg):
    if framework_name in ["early_fusion", "early_fusion_mat"]:
        n_heads = cfg.framework.attention_n_heads
        if cfg.framework.attention_name == "self_attention":
            attention_modules = SelfAttention(framework_name, feat_sizes, cfg.framework.attention_layers, n_heads)
        # elif cfg.framework.attention_name == "axial_attention":
        #     attention_modules = SelfAxialAttention(framework_name, feat_sizes, cfg.framework.attention_layers, n_heads)
        elif cfg.framework.attention_name == "pymd_self_attention":
            attention_modules = AsymmetricPyramidSelfAttention(
                framework_name, feat_sizes, cfg.framework.attention_layers, n_heads
            )
        elif cfg.framework.attention_name is None:
            attention_modules = None
        else:
            raise NotImplementedError(
                f'Attention mode "{cfg.framework.attention_name}" call for "{framework_name}" framework '
            )

    elif framework_name == "late_fusion":
        n_heads = cfg.framework.attention_n_heads
        assert (
            cfg.framework.sequence_name != "patch_transformer"
        ), "Attention not compatible with patch transformer features."
        if cfg.framework.attention_name == "self_attention":
            attention_modules = SelfAttention(framework_name, feat_sizes, cfg.framework.attention_layers, n_heads)
        # elif cfg.framework.attention_name == "axial_attention":
        #     attention_modules = SelfAxialAttention(framework_name, feat_sizes, cfg.framework.attention_layers, n_heads)
        elif cfg.framework.attention_name == "pymd_self_attention":
            attention_modules = AsymmetricPyramidSelfAttention(
                framework_name, feat_sizes, cfg.framework.attention_layers, n_heads
            )
        elif cfg.framework.attention_name is None:
            attention_modules = None
        else:
            raise NotImplementedError(
                f'Attention mode "{cfg.framework.attention_name}" call for "{framework_name}" framework '
            )
    elif framework_name in ["patch_transformer", "attn_space_time"]:
        attention_modules = None
    else:
        raise NotImplementedError(f"Attention for framework name {framework_name} not implemented.")

    return attention_modules
