# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import math
import os
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
#torch.cuda.set_device(1)
from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C3,
    C3TR,
    OBB,
    SPP,
    SPPELAN,
    SPPF,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2PSA,
    C2fAttn,
    C3Ghost,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Pose,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    ResNetLayer,
    RTDETRDecoder,
    Segment,
    Silence,
    WorldDetect,
    Concat2,
    ADD,
    ShuffleAttention,
    SimAM,
    GAM_Attention,
    CBAM2,
    CoordAtt,
    ECA,
    SEAttention,
    GLCBAM,
    S2Attention,
    SKAttention,
    GLF,
    NAM,
    GCBAM,
    SACBAM,
    MdC2f,
    CDC2f,
    C2f_Shufflenet,
    C2f_Invo,
    C2f_PKIModule,
    CSFM,
    FEM,
    C2f_FEM,
    C2f_PPA,
    C2f_RG,
    C2f_Faster,
    Fusion,
    Concat3,
    RIFusion,
    frequencyfuse,
    fusehigh,
    fuselow,
    fusemiddle,
    Cross_modal_Vmaba_module2,
    ChannelAttention2,
    SpatialAttention2,
    CBAM2,
    Add,
    Add2,
    myTransformerBlock,
    IRRGBGatedFusion,
    TalorNet,
    AdapterTalorNet,
    P3AdapterTalorNet,
    PixelTaylorFusion,
    HeadTaylorDetailInjector,
    getreturntalorfuse,
    getreturntalorir,
    fusehd_second,
    fusehd_second_optimized,
    fusehd_third_optimized,
    fusehd_fourth,
    spatial_fuse_ncssd,
    VMAMBA2BasicLayer,
    getreturnir,
    getreturnrgb,
    getreturnfeature,
    getreturnfeature2,
    getreturnfuse,
    ResNetLayer2,
    getreturnfhigh,
    returnlowir,
    returnlowrgb,
    returnfeature,
    VSSLayer,
    A2C2f,
    C3k2,
    AuxDetect,
    P2BidirectionalAdaptiveInjection,
    P2RGBCorrectionOnly,
    P2RGBAnchorProtectedInjection,
    P2FrequencySelectiveFusion,
    PretrainedAnchorP3Fusion,
)
from ultralytics.nn.modules.scale_assignment import P2BidirectionalTaylorExchange, P3DetailCorrection
from ultralytics.nn.modules.single_scale_fusion import RGBPassthrough
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import v8ClassificationLoss, v8DetectionLoss, v8OBBLoss, v8PoseLoss, v8SegmentationLoss
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    make_divisible,
    model_info,
    scale_img,
    time_sync,
)

try:
    import thop
except ImportError:
    thop = None


class BaseModel(nn.Module):
    """The BaseModel class serves as a base class for all the models in the Ultralytics YOLO family."""

    def forward(self, x, *args, **kwargs):
        """
        Forward pass of the model on a single scale. Wrapper for `_forward_once` method.

        Args:
            x (torch.Tensor | dict): The input image tensor or a dict including image tensor and gt labels.

        Returns:
            (torch.Tensor): The output of the network.
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            augment (bool): Augment image during prediction, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        if getattr(self, "pretrained_anchor_config", None) is not None:
            return self._predict_pretrained_anchor(x, profile, visualize, embed)
        y, dt, embeddings = [], [], []  # outputs
        pixel_fusion = getattr(self, "pixel_fusion", None)
        single_stream = pixel_fusion is not None
        if single_stream:
            # A six-channel aligned pair becomes one three-channel task image
            # before the unmodified official YOLO backbone.
            x = pixel_fusion(x)
            head_taylor_detail = pixel_fusion.pop_head_taylor_detail()
            fusion_aux_head = getattr(self, "fusion_aux_head", None)
            if self.training and fusion_aux_head is not None and fusion_aux_head.loss_weight > 0.0:
                fusion_aux_head([self.fusion_aux_encoder(x)])
            elif fusion_aux_head is not None:
                fusion_aux_head.last_preds = None
            rgb = ir = None
        else:
            head_taylor_detail = None
            rgb,ir=torch.chunk(x,chunks=2,dim=1) # ����
            # rgb=x[:, :3, :, :] # �ɼ���
            x=rgb
        # if(x.shape[1]==6) :
        #     # import matplotlib.pyplot as plt
        #     # import cv2
        #     # import numpy as np
        #     # img1= x[0][:3,...] #rgb
        #     # img2= x[0][3:,...] #ir
        #     # img1=img1.permute(1, 2, 0)
        #     # img2=img2.permute(1,2,0)
        #     # if type(img2==torch.Tensor):
        #     #     img2=img2.cpu()
        #     #     img2=np.array(img2)
        #     # else :
        #     #     img2=np.array(img2)
        #     # if type(img1==torch.Tensor):
        #     #     img1=img1.cpu()
        #     # plt.imshow(img1)
        #     # plt.savefig('/home/mjy/ultralytics/images/'+str(1)+'rgb.jpg')
        #     # plt.close()
        #     # plt.imshow(img2)
        #     # plt.savefig('/home/mjy/ultralytics/images/'+str(1)+'ir.jpg')
        #     # plt.close()
        #     rgb,ir=torch.chunk(x,chunks=2,dim=1) # ����
        #     # rgb=x[:, :3, :, :] # �ɼ���
        #     x=rgb
        # else :
        #     rgb=x
        #     ir=x


        isR=False

        for m in self.model:


            if m.f != -4:
                if m.f != -3:
                    if m.f != -1:  # if not from previous layer
                        x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                self._profile_one_layer(m, x, dt)



            if single_stream:
                if m is self.model[-1] and head_taylor_detail is not None:
                    x = self.pixel_taylor_head_injector(x, head_taylor_detail)
                x = m(x)
            elif m.f==-4:
                if isR:
                    x= m(ir)
                    ir=x
                    isR=False
                else :
                    x = m(rgb)  # run
                    rgb=x
                    isR=True
            elif m.f==-3:
                    if(len(list(m.named_parameters()))>0):
                        x3=torch.cat([rgb,ir],dim=1)
                        x3=m(x3)
                        rgb,ir = torch.chunk(x3, 2, dim=1)
                        x=[]
            else :
                x=m(x)
            """elif m.i<23:
                if isR:
                    x= m(rgb)
                    rgb=x
                else :
                    x = m(ir)
                    ir=x"""




            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)

        return x

    def _predict_pretrained_anchor(self, x, profile=False, visualize=False, embed=None):
        """Run a standard YOLO stream or fuse independently-normalized RGB/IR P3 features."""
        if x.ndim != 4 or x.shape[1] != 6:
            raise ValueError(f"Pretrained-anchor model expects BCHW input with 6 channels, got {tuple(x.shape)}")
        config = self.pretrained_anchor_config
        mode = config["mode"]
        rgb, ir = torch.chunk(x, 2, dim=1)
        y, dt, embeddings = [], [], []
        if mode in {"rgb_only", "ir_only"}:
            x = rgb if mode == "rgb_only" else ir
        else:
            fusion_layer = int(config["fusion_layer"])
            deep_fusions = getattr(self, "anchor_deep_taylor_fusions", None)
            deep_layers = tuple(int(index) for index in deep_fusions.keys()) if deep_fusions is not None else ()
            last_dual_layer = 2 if mode == "p2_only" else max((fusion_layer, *deep_layers))

        for m in self.model:
            if mode not in {"rgb_only", "ir_only"} and m.i <= last_dual_layer:
                if profile:
                    self._profile_one_layer(m, rgb, dt)
                rgb = m(rgb)
                ir = self.anchor_ir_stem[m.i](ir)
                p2_bidirectional = getattr(self, "anchor_p2_bidirectional_injection", None)
                if (
                    p2_bidirectional is not None
                    and m.i == self.anchor_p2_bidirectional_config["fusion_layer"]
                ):
                    rgb, ir = p2_bidirectional(rgb, ir)
                p2_rgb_only = getattr(self, "anchor_p2_rgb_only_injection", None)
                if (
                    p2_rgb_only is not None
                    and m.i == self.anchor_p2_bidirectional_config["fusion_layer"]
                ):
                    rgb = p2_rgb_only(rgb, ir)
                p2_rgb_protected = getattr(self, "anchor_p2_rgb_protected_injection", None)
                if (
                    p2_rgb_protected is not None
                    and m.i == self.anchor_p2_rgb_protected_config["fusion_layer"]
                ):
                    rgb, ir = p2_rgb_protected(rgb, ir)
                p2_fusion = getattr(self, "anchor_p2_detail_fusion", None)
                if p2_fusion is not None and m.i == self.anchor_p2_aux_config["fusion_layer"]:
                    aux_head = self.fusion_aux_head
                    # Keep this branch in the DDP autograd graph even after its
                    # scheduled weight reaches zero.  Skipping it at the decay
                    # boundary makes previously-used parameters become unused
                    # mid-run and DDP aborts before the next epoch.
                    if self.training:
                        aux_head([p2_fusion(rgb, ir)])
                    else:
                        aux_head.last_preds = None
                if m.i == fusion_layer:
                    rgb = self.anchor_p3_fusion(rgb, ir)
                elif deep_fusions is not None and str(m.i) in deep_fusions:
                    rgb = deep_fusions[str(m.i)](rgb, ir)
                x = rgb
            else:
                if m.f != -1:
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
                if profile:
                    self._profile_one_layer(m, x, dt)
                x = m(x)
            y.append(x if m.i in self.save else None)
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(
            f"WARNING ⚠️ {self.__class__.__name__} does not support augmented inference yet. "
            f"Reverting to single-scale inference instead."
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """
        Profile the computation time and FLOPs of a single layer of the model on a given input. Appends the results to
        the provided list.

        Args:
            m (nn.Module): The layer to be profiled.
            x (torch.Tensor): The input data to the layer.
            dt (list): A list to store the computation time of the layer.

        Returns:
            None
        """
        c = m == self.model[-1] and isinstance(x, list)  # is final layer list, copy input as inplace fix
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer, in order to improve the
        computation efficiency.

        Returns:
            (nn.Module): The fused model is returned.
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # update forward
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """
        Check if the model has less than a certain threshold of BatchNorm layers.

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers. Default is 10.

        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

    def info(self, detailed=False, verbose=True, imgsz=640):
        """
        Prints model information.

        Args:
            detailed (bool): if True, prints out detailed information about the model. Defaults to False
            verbose (bool): if True, prints out the model information. Defaults to False
            imgsz (int): the size of the image that the model will be trained on. Defaults to 640
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """
        Applies a function to all the tensors in the model that are not parameters or registered buffers.

        Args:
            fn (function): the function to apply to the model

        Returns:
            (BaseModel): An updated BaseModel object.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """
        Load the weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress. Defaults to True.
        """
        model = weights["model"] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(csd, strict=False)  # load
        if verbose:
            LOGGER.info(f"Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights")

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        preds = self.forward(batch["img"]) if preds is None else preds
        return self.criterion(preds, batch)

    def init_criterion(self):
        """Initialize the loss criterion for the BaseModel."""
        raise NotImplementedError("compute_loss() needs to be implemented by task heads")


class DetectionModel(BaseModel):
    """YOLOv8 detection model."""

    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        """Initialize the YOLOv8 detection model with the given config and parameters."""
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # loader input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value

        pixel_fusion_config = self.yaml.get("pixel_fusion")
        pretrained_anchor_config = self.yaml.get("pretrained_anchor_fusion")
        if pixel_fusion_config is not None and pretrained_anchor_config is not None:
            raise ValueError("pixel_fusion and pretrained_anchor_fusion are mutually exclusive")
        if pixel_fusion_config is not None:
            if ch != 6:
                raise ValueError(f"Pixel fusion model YAML must declare ch: 6, got ch={ch}")
            if not isinstance(pixel_fusion_config, dict):
                raise TypeError("model.yaml pixel_fusion must be a mapping")
            self.pixel_fusion = PixelTaylorFusion(**deepcopy(pixel_fusion_config))
            backbone_ch = 3
        elif pretrained_anchor_config is not None:
            if ch != 6:
                raise ValueError(f"Pretrained-anchor model YAML must declare ch: 6, got ch={ch}")
            if not isinstance(pretrained_anchor_config, dict):
                raise TypeError("model.yaml pretrained_anchor_fusion must be a mapping")
            backbone_ch = 3
        else:
            backbone_ch = ch
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=backbone_ch, verbose=verbose)  # model, savelist
        if pretrained_anchor_config is not None:
            anchor_mode = str(pretrained_anchor_config.get("mode", "residual"))
            if anchor_mode not in {"rgb_only", "ir_only", "p2_only", "mean", "residual"}:
                raise ValueError(f"Unsupported pretrained-anchor mode: {anchor_mode}")
            fusion_layer = int(pretrained_anchor_config.get("fusion_layer", 4))
            if fusion_layer != 4:
                raise ValueError("The first pretrained-anchor experiment supports P3 layer 4 only")
            self.pretrained_anchor_config = {
                "mode": anchor_mode,
                "anchor_modality": str(pretrained_anchor_config.get("anchor_modality", "rgb")),
                "fusion_layer": fusion_layer,
                "order": int(pretrained_anchor_config.get("order", 1)),
                "recurrence": str(pretrained_anchor_config.get("recurrence", "additive")),
            }
            if self.pretrained_anchor_config["anchor_modality"] not in {"rgb", "ir"}:
                raise ValueError("Pretrained-anchor P3 anchor_modality must be rgb or ir")
            anchor_deep_taylor_config = self.yaml.get("anchor_deep_taylor_fusions")
            if anchor_deep_taylor_config is not None:
                if not isinstance(anchor_deep_taylor_config, dict):
                    raise TypeError("anchor_deep_taylor_fusions must be a mapping")
                if anchor_mode != "residual" or self.pretrained_anchor_config["anchor_modality"] != "rgb":
                    raise ValueError("Deep Taylor fusion requires residual mode with an RGB P3 anchor")
                deep_layers = tuple(dict.fromkeys(int(index) for index in anchor_deep_taylor_config.get("layers", ())))
                if not deep_layers or any(index not in {6, 9} for index in deep_layers):
                    raise ValueError("Deep Taylor fusion layers must be a non-empty subset of YOLOv8 P4=6 and P5=9")
                if tuple(sorted(deep_layers)) != deep_layers:
                    raise ValueError("Deep Taylor fusion layers must be ordered from shallow to deep")
            else:
                deep_layers = ()
            if anchor_mode == "p2_only":
                self.anchor_ir_stem = deepcopy(self.model[:3])
                reference = self.yaml.get("p2_only_reference_p3_init")
                if not isinstance(reference, dict):
                    raise ValueError("P2-only mode requires p2_only_reference_p3_init metadata")
                fusion_source = self.model[fusion_layer]
                if not isinstance(fusion_source, C2f):
                    raise TypeError("P2-only reference initialization expects layer 4 to be C2f")
                # The full model constructs P3 O3 immediately before P2. Build
                # and discard the same block so retained P2 tensors have the
                # same initialization for each paired seed.
                _discarded_reference_p3 = PretrainedAnchorP3Fusion(
                    fusion_source.cv2.conv.out_channels,
                    mode="residual",
                    hidden=int(reference.get("hidden", 16)),
                    gate_max=float(reference.get("gate_max", 0.25)),
                    order=int(reference.get("order", 3)),
                    recurrence=str(reference.get("recurrence", "additive")),
                    anchor_modality="rgb",
                )
                del _discarded_reference_p3
            if anchor_mode in {"mean", "residual"}:
                fusion_source = self.model[fusion_layer]
                if not isinstance(fusion_source, C2f):
                    raise TypeError("P3 pretrained-anchor fusion expects layer 4 to be C2f")
                anchor_channels = fusion_source.cv2.conv.out_channels
                last_dual_layer = max((fusion_layer, *deep_layers))
                self.anchor_ir_stem = deepcopy(self.model[: last_dual_layer + 1])
                self.anchor_p3_fusion = PretrainedAnchorP3Fusion(
                    anchor_channels,
                    mode=anchor_mode,
                    hidden=int(pretrained_anchor_config.get("hidden", 16)),
                    gate_max=float(pretrained_anchor_config.get("gate_max", 0.25)),
                    order=int(pretrained_anchor_config.get("order", 1)),
                    recurrence=str(pretrained_anchor_config.get("recurrence", "additive")),
                    anchor_modality=self.pretrained_anchor_config["anchor_modality"],
                )
                if deep_layers:
                    self.anchor_deep_taylor_config = {
                        "layers": deep_layers,
                        "hidden": int(anchor_deep_taylor_config.get("hidden", 16)),
                        "gate_max": float(anchor_deep_taylor_config.get("gate_max", 0.25)),
                        "order": int(anchor_deep_taylor_config.get("order", 3)),
                        "recurrence": str(anchor_deep_taylor_config.get("recurrence", "additive")),
                    }
                    self.anchor_deep_taylor_fusions = nn.ModuleDict()
                    for layer in deep_layers:
                        source = self.model[layer]
                        if isinstance(source, C2f):
                            channels = source.cv2.conv.out_channels
                        elif isinstance(source, SPPF):
                            channels = source.cv2.conv.out_channels
                        else:
                            raise TypeError(f"Deep Taylor fusion layer {layer} must be C2f or SPPF")
                        self.anchor_deep_taylor_fusions[str(layer)] = PretrainedAnchorP3Fusion(
                            channels,
                            mode="residual",
                            hidden=self.anchor_deep_taylor_config["hidden"],
                            gate_max=self.anchor_deep_taylor_config["gate_max"],
                            order=self.anchor_deep_taylor_config["order"],
                            recurrence=self.anchor_deep_taylor_config["recurrence"],
                            anchor_modality="rgb",
                        )
        aux_stride = None
        anchor_p2_bidirectional_config = self.yaml.get("anchor_p2_bidirectional_fusion")
        if anchor_p2_bidirectional_config is not None:
            if pretrained_anchor_config is None:
                raise ValueError("anchor_p2_bidirectional_fusion requires pretrained_anchor_fusion")
            if not isinstance(anchor_p2_bidirectional_config, dict):
                raise TypeError("model.yaml anchor_p2_bidirectional_fusion must be a mapping")
            if str(pretrained_anchor_config.get("mode", "residual")) not in {"residual", "p2_only"}:
                raise ValueError("anchor_p2_bidirectional_fusion requires residual P3 fusion or P2-only mode")
            p2_layer = int(anchor_p2_bidirectional_config.get("fusion_layer", 2))
            if p2_layer != 2 or not isinstance(self.model[p2_layer], C2f):
                raise ValueError("P2 bidirectional injection requires C2f backbone layer 2")
            p2_channels = self.model[p2_layer].cv2.conv.out_channels
            self.anchor_p2_bidirectional_config = {
                "mode": "bidirectional_adaptive",
                "fusion_layer": p2_layer,
                "hidden": int(anchor_p2_bidirectional_config.get("hidden", 16)),
                "gate_max": float(anchor_p2_bidirectional_config.get("gate_max", 0.10)),
            }
            if anchor_mode == "p2_only":
                self.anchor_p2_rgb_only_injection = P2RGBCorrectionOnly(
                    p2_channels,
                    hidden=self.anchor_p2_bidirectional_config["hidden"],
                    gate_max=self.anchor_p2_bidirectional_config["gate_max"],
                )
            else:
                self.anchor_p2_bidirectional_injection = P2BidirectionalAdaptiveInjection(
                    p2_channels,
                    hidden=self.anchor_p2_bidirectional_config["hidden"],
                    gate_max=self.anchor_p2_bidirectional_config["gate_max"],
                )
        scale_assignment = self.yaml.get("anchor_scale_assignment")
        if scale_assignment is not None:
            tag = scale_assignment.get("assignment")
            if tag not in {"DT", "DD", "TT", "TD"}:
                raise ValueError("Scale assignment must be DT, DD, TT or TD")
            if (anchor_p2_bidirectional_config is None or anchor_mode != "residual"
                    or self.pretrained_anchor_config["anchor_modality"] != "rgb"
                    or deep_layers or self.yaml.get("anchor_p2_aux_fusion") is not None
                    or self.yaml.get("anchor_p2_rgb_protected_fusion") is not None):
                raise ValueError("Scale assignment requires P2 bidirectional + RGB P3 residual, without extra fusions")
            if self.pretrained_anchor_config["order"] != 3 or self.pretrained_anchor_config["recurrence"] != "additive":
                raise ValueError("Scale assignment fixes Taylor order=3 and additive recurrence")
            self.anchor_scale_assignment_config = deepcopy(scale_assignment)
            # Independent role-specific RNG streams keep shared modules identical
            # across the factorial combinations; DT remains bitwise legacy-compatible.
            init_seed = int(scale_assignment.get("init_seed", 0))
            if tag[1] == "D":
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(init_seed + 1003)
                    self.anchor_p3_fusion = P3DetailCorrection(
                        anchor_channels, hidden=int(pretrained_anchor_config.get("hidden", 16)),
                        gate_max=float(pretrained_anchor_config.get("gate_max", 0.25)),
                    )
            if tag[0] == "T":
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(init_seed + 1002)
                    self.anchor_p2_bidirectional_injection = P2BidirectionalTaylorExchange(
                        p2_channels, hidden=self.anchor_p2_bidirectional_config["hidden"],
                        gate_max=self.anchor_p2_bidirectional_config["gate_max"], order=3,
                    )
                self.anchor_p2_bidirectional_config["mode"] = "bidirectional_taylor"
        single_scale = self.yaml.get("anchor_single_taylor_scale")
        if single_scale is not None:
            scale = single_scale.get("scale")
            if scale not in {"p3", "p4", "p5"}:
                raise ValueError("Single Taylor scale must be p3, p4 or p5")
            if (anchor_p2_bidirectional_config is None or anchor_mode != "residual"
                    or self.pretrained_anchor_config["anchor_modality"] != "rgb"
                    or deep_layers or scale_assignment is not None
                    or self.yaml.get("anchor_p2_aux_fusion") is not None
                    or self.yaml.get("anchor_p2_rgb_protected_fusion") is not None
                    or self.pretrained_anchor_config["order"] != 3
                    or self.pretrained_anchor_config["recurrence"] != "additive"):
                raise ValueError("Single-scale O3 requires fixed bidirectional P2 and no other fusion variants")
            self.anchor_single_taylor_config = deepcopy(single_scale)
            if scale != "p3":
                layer = {"p4": 6, "p5": 9}[scale]
                source = self.model[layer]
                if not isinstance(source, (C2f, SPPF)):
                    raise TypeError("Single-scale Taylor expects P4 C2f or P5 SPPF")
                # Construct AFTER the legacy P2 module: P2 and shared body states
                # remain bitwise identical to the original P3-only reference.
                self.anchor_ir_stem = deepcopy(self.model[: layer + 1])
                self.anchor_p3_fusion = RGBPassthrough()
                self.anchor_deep_taylor_config = {
                    "layers": (layer,), "hidden": int(pretrained_anchor_config.get("hidden", 16)),
                    "gate_max": float(pretrained_anchor_config.get("gate_max", .25)),
                    "order": 3, "recurrence": "additive",
                }
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(int(single_scale.get("init_seed", 0)) + 2000 + layer)
                    self.anchor_deep_taylor_fusions = nn.ModuleDict({str(layer): PretrainedAnchorP3Fusion(
                        source.cv2.conv.out_channels, mode="residual",
                        hidden=self.anchor_deep_taylor_config["hidden"],
                        gate_max=self.anchor_deep_taylor_config["gate_max"],
                        order=3, recurrence="additive", anchor_modality="rgb",
                    )})
        anchor_p2_rgb_protected_config = self.yaml.get("anchor_p2_rgb_protected_fusion")
        if anchor_p2_rgb_protected_config is not None:
            if pretrained_anchor_config is None:
                raise ValueError("anchor_p2_rgb_protected_fusion requires pretrained_anchor_fusion")
            if anchor_p2_bidirectional_config is not None:
                raise ValueError("P2 bidirectional and RGB-protected injections are separate experiments")
            if not isinstance(anchor_p2_rgb_protected_config, dict):
                raise TypeError("model.yaml anchor_p2_rgb_protected_fusion must be a mapping")
            if str(pretrained_anchor_config.get("mode", "residual")) != "residual":
                raise ValueError("anchor_p2_rgb_protected_fusion requires residual P3 anchor fusion")
            if str(pretrained_anchor_config.get("anchor_modality", "rgb")) != "rgb":
                raise ValueError("P2 RGB-anchor protection requires an RGB-anchored P3 fusion")
            p2_layer = int(anchor_p2_rgb_protected_config.get("fusion_layer", 2))
            if p2_layer != 2 or not isinstance(self.model[p2_layer], C2f):
                raise ValueError("P2 RGB-anchor protection requires a C2f-compatible backbone layer 2")
            p2_channels = self.model[p2_layer].cv2.conv.out_channels
            self.anchor_p2_rgb_protected_config = {
                "mode": "rgb_anchor_protected",
                "fusion_layer": p2_layer,
                "hidden": int(anchor_p2_rgb_protected_config.get("hidden", 16)),
                "gate_max": float(anchor_p2_rgb_protected_config.get("gate_max", 0.10)),
            }
            self.anchor_p2_rgb_protected_injection = P2RGBAnchorProtectedInjection(
                p2_channels,
                hidden=self.anchor_p2_rgb_protected_config["hidden"],
                gate_max=self.anchor_p2_rgb_protected_config["gate_max"],
            )
        anchor_p2_aux_config = self.yaml.get("anchor_p2_aux_fusion")
        if anchor_p2_aux_config is not None:
            if anchor_p2_bidirectional_config is not None or anchor_p2_rgb_protected_config is not None:
                raise ValueError("P2 main-path injection and P2 auxiliary fusion are separate experiments")
            if pretrained_anchor_config is None:
                raise ValueError("anchor_p2_aux_fusion requires pretrained_anchor_fusion")
            if not isinstance(anchor_p2_aux_config, dict):
                raise TypeError("model.yaml anchor_p2_aux_fusion must be a mapping")
            if str(pretrained_anchor_config.get("mode", "residual")) != "residual":
                raise ValueError("anchor_p2_aux_fusion requires residual P3 anchor fusion")
            p2_layer = int(anchor_p2_aux_config.get("fusion_layer", 2))
            if p2_layer != 2 or not isinstance(self.model[p2_layer], C2f):
                raise ValueError("The P2 auxiliary experiment requires C2f backbone layer 2")
            p2_channels = self.model[p2_layer].cv2.conv.out_channels
            aux_stride = 4
            self.anchor_p2_aux_config = {
                "mode": "multifilter",
                "fusion_layer": p2_layer,
                "stride": aux_stride,
            }
            self.anchor_p2_detail_fusion = P2FrequencySelectiveFusion(
                p2_channels,
                hidden=int(anchor_p2_aux_config.get("hidden", 16)),
                gate_max=float(anchor_p2_aux_config.get("gate_max", 0.10)),
            )
            self.fusion_aux_head = AuxDetect(
                nc=int(self.yaml["nc"]),
                loss_weight=float(anchor_p2_aux_config.get("loss_weight", 0.10)),
                ch=(p2_channels,),
                decay_start_epoch=int(anchor_p2_aux_config.get("decay_start_epoch", -1)),
                decay_end_epoch=int(anchor_p2_aux_config.get("decay_end_epoch", -1)),
            )
        fusion_aux_config = self.yaml.get("fusion_aux_detect")
        if fusion_aux_config is not None:
            if pixel_fusion_config is None:
                raise ValueError("fusion_aux_detect requires pixel_fusion")
            if not isinstance(fusion_aux_config, dict):
                raise TypeError("model.yaml fusion_aux_detect must be a mapping")
            aux_channels = int(fusion_aux_config.get("channels", 64))
            aux_stride = int(fusion_aux_config.get("stride", 4))
            aux_weight = float(fusion_aux_config.get("loss_weight", 0.2))
            aux_decay_start = int(fusion_aux_config.get("decay_start_epoch", -1))
            aux_decay_end = int(fusion_aux_config.get("decay_end_epoch", -1))
            if aux_stride not in {4, 8, 16}:
                raise ValueError(f"fusion_aux_detect stride must be 4, 8, or 16, got {aux_stride}")
            encoder = [
                Conv(3, aux_channels // 2, 3, 2),
                Conv(aux_channels // 2, aux_channels, 3, 2),
            ]
            for _ in range(int(math.log2(aux_stride)) - 2):
                encoder.append(Conv(aux_channels, aux_channels, 3, 2))
            encoder.append(C2f(aux_channels, aux_channels, n=1, shortcut=True))
            self.fusion_aux_encoder = nn.Sequential(*encoder)
            self.fusion_aux_head = AuxDetect(
                nc=int(self.yaml["nc"]),
                loss_weight=aux_weight,
                ch=(aux_channels,),
                decay_start_epoch=aux_decay_start,
                decay_end_epoch=aux_decay_end,
            )
        if pixel_fusion_config is not None and pixel_fusion_config.get("mode") == "head_taylor":
            detect = self.model[-1]
            if not isinstance(detect, Detect):
                raise TypeError("head_taylor requires a Detect layer at the end of the model")
            p3_channels = detect.cv2[0][0].conv.in_channels
            self.pixel_taylor_head_injector = HeadTaylorDetailInjector(
                p3_channels,
                hidden=int(pixel_fusion_config.get("hidden", 8)) * 2,
                gate_max=float(pixel_fusion_config.get("gate_max", 0.1)),
            )
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
            s = 640  # 2x min stride
            m.inplace = self.inplace

            forward = lambda x: self.forward(x)[0] if isinstance(m, (Segment, Pose, OBB)) else self.forward(x)
            #m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(2, 6, s, s))])  # forward
            probe_override = os.environ.get("ULTRALYTICS_MODEL_PROBE_DEVICE")
            probe_device = torch.device(probe_override or ("cuda" if torch.cuda.is_available() else "cpu"))
            self.model.to(probe_device)
            if getattr(self, "pixel_fusion", None) is not None:
                self.pixel_fusion.to(probe_device)
            if getattr(self, "pixel_taylor_head_injector", None) is not None:
                # This side branch is registered directly on DetectionModel rather
                # than inside self.model, so the stride probe must move it explicitly.
                self.pixel_taylor_head_injector.to(probe_device)
            if getattr(self, "fusion_aux_encoder", None) is not None:
                self.fusion_aux_encoder.to(probe_device)
                self.fusion_aux_head.to(probe_device)
            if getattr(self, "anchor_ir_stem", None) is not None:
                self.anchor_ir_stem.to(probe_device)
                if getattr(self, "anchor_p3_fusion", None) is not None:
                    self.anchor_p3_fusion.to(probe_device)
            if getattr(self, "anchor_deep_taylor_fusions", None) is not None:
                self.anchor_deep_taylor_fusions.to(probe_device)
            if getattr(self, "anchor_p2_detail_fusion", None) is not None:
                self.anchor_p2_detail_fusion.to(probe_device)
                self.fusion_aux_head.to(probe_device)
            if getattr(self, "anchor_p2_bidirectional_injection", None) is not None:
                self.anchor_p2_bidirectional_injection.to(probe_device)
            if getattr(self, "anchor_p2_rgb_only_injection", None) is not None:
                self.anchor_p2_rgb_only_injection.to(probe_device)
            if getattr(self, "anchor_p2_rgb_protected_injection", None) is not None:
                self.anchor_p2_rgb_protected_injection.to(probe_device)
            m.stride = torch.tensor(
                [s / x.shape[-2] for x in forward(torch.zeros(2, 6, s, s, device=probe_device))]
            )
            self.stride = m.stride
            m.bias_init()  # only run once
            for aux_head in (layer for layer in self.model if isinstance(layer, AuxDetect)):
                aux_head.stride = m.stride.clone()
                aux_head.bias_init()
                # The stride probe runs in training mode, so AuxDetect caches
                # graph-connected predictions. They are temporary build data,
                # not model state, and current PyTorch cannot deepcopy them
                # when profiling or constructing ModelEMA.
                aux_head.last_preds = None
            fusion_aux_head = getattr(self, "fusion_aux_head", None)
            if fusion_aux_head is not None:
                if aux_stride is None:
                    raise RuntimeError("Fusion auxiliary head is missing its stride")
                fusion_aux_head.stride = torch.tensor([float(aux_stride)], device=probe_device)
                fusion_aux_head.bias_init()
                fusion_aux_head.last_preds = None
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")




    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs."""
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails."""
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return v8DetectionLoss(self)

    def loss(self, batch, preds=None):
        """Compute the original v1 main loss plus the IR zero-order auxiliary-head loss."""
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()
        pixel_fusion = getattr(self, "pixel_fusion", None)
        owns_forward = preds is None
        if owns_forward:
            if pixel_fusion is not None:
                pixel_fusion.collect_auxiliary_state = True
            try:
                preds = self.forward(batch["img"])
            finally:
                if pixel_fusion is not None:
                    pixel_fusion.collect_auxiliary_state = False
        w_rgb = batch.get("loss_rgb_weight", 1.0)
        w_ir = batch.get("loss_ir_weight", 1.0)

        def branch_loss(branch_preds, criterion=None):
            criterion = self.criterion if criterion is None else criterion
            loss_rgb, items = criterion(branch_preds, batch)
            total = w_rgb * loss_rgb
            weighted_items = w_rgb * items
            if "cls_ir" in batch and batch["cls_ir"].numel() > 0:
                dev = batch["bboxes"].device
                batch_ir = {
                    "batch_idx": batch["batch_idx_ir"].view(-1, 1).to(device=dev, dtype=torch.long),
                    "cls": (
                        batch["cls_ir"].view(-1, 1).to(device=dev, dtype=torch.long)
                        if batch["cls_ir"].dim() < 2
                        else batch["cls_ir"].to(device=dev, dtype=torch.long)
                    ),
                    "bboxes": batch["bboxes_ir"].to(device=dev, dtype=batch["bboxes"].dtype),
                }
                loss_ir, items_ir = criterion(branch_preds, batch_ir)
                total = total + w_ir * loss_ir
                weighted_items = weighted_items + w_ir * items_ir
            return total, weighted_items

        total_loss, loss_items = branch_loss(preds)
        if (
            pixel_fusion is not None
            and pixel_fusion.has_auxiliary_loss
            and self.training
            and owns_forward
        ):
            # Detection loss in this Ultralytics version is batch-summed. Keep
            # image regularizers on the same scale before applying their small
            # explicit weights inside PixelTaylorFusion.
            total_loss = total_loss + batch["img"].shape[0] * pixel_fusion.auxiliary_loss(batch)
        aux_head = next((layer for layer in self.model if isinstance(layer, AuxDetect)), None)
        if aux_head is not None and aux_head.last_preds is not None:
            aux_loss, aux_items = branch_loss(aux_head.last_preds)
            total_loss = total_loss + aux_head.loss_weight * aux_loss
            loss_items = loss_items + aux_head.loss_weight * aux_items
        fusion_aux_head = getattr(self, "fusion_aux_head", None)
        if fusion_aux_head is not None and fusion_aux_head.last_preds is not None:
            if not hasattr(self, "fusion_aux_criterion"):
                self.fusion_aux_criterion = v8DetectionLoss(self)
                self.fusion_aux_criterion.stride = fusion_aux_head.stride
            aux_loss, _ = branch_loss(fusion_aux_head.last_preds, self.fusion_aux_criterion)
            fusion_aux_head.set_loss_diagnostics(aux_loss, total_loss)
            total_loss = total_loss + fusion_aux_head.loss_weight * aux_loss
            # Release this full-resolution graph immediately after its one consumer.
            fusion_aux_head.last_preds = None
        return total_loss, loss_items


class OBBModel(DetectionModel):
    """YOLOv8 Oriented Bounding Box (OBB) model."""

    def __init__(self, cfg="yolov8n-obb.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 OBB model with given config and parameters."""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)


    def init_criterion(self):
        """Initialize the loss criterion for the model."""
        return v8OBBLoss(self)


class SegmentationModel(DetectionModel):
    """YOLOv8 segmentation model."""

    def __init__(self, cfg="yolov8n-seg.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 segmentation model with given config and parameters."""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the SegmentationModel."""
        return v8SegmentationLoss(self)


class PoseModel(DetectionModel):
    """YOLOv8 pose model."""

    def __init__(self, cfg="yolov8n-pose.yaml", ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """Initialize YOLOv8 Pose model."""
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)  # load model YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg["kpt_shape"] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the PoseModel."""
        return v8PoseLoss(self)


class ClassificationModel(BaseModel):
    """YOLOv8 classification model."""

    def __init__(self, cfg="yolov8n-cls.yaml", ch=3, nc=None, verbose=True):
        """Init ClassificationModel with YAML, channels, number of classes, verbose flag."""
        super().__init__()
        self._from_yaml(cfg, ch, nc, verbose)

    def _from_yaml(self, cfg, ch, nc, verbose):
        """Set YOLOv8 model configurations and define the model architecture."""
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        elif not nc and not self.yaml.get("nc", None):
            raise ValueError("nc not specified. Must specify nc in model.yaml or function arguments.")
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.stride = torch.Tensor([1])  # no stride constraints
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.info()

    @staticmethod
    def reshape_outputs(model, nc):
        """Update a TorchVision classification model to class count 'n' if required."""
        name, m = list((model.model if hasattr(model, "model") else model).named_children())[-1]  # last module
        if isinstance(m, Classify):  # YOLO Classify() head
            if m.linear.out_features != nc:
                m.linear = nn.Linear(m.linear.in_features, nc)
        elif isinstance(m, nn.Linear):  # ResNet, EfficientNet
            if m.out_features != nc:
                setattr(model, name, nn.Linear(m.in_features, nc))
        elif isinstance(m, nn.Sequential):
            types = [type(x) for x in m]
            if nn.Linear in types:
                i = types.index(nn.Linear)  # nn.Linear index
                if m[i].out_features != nc:
                    m[i] = nn.Linear(m[i].in_features, nc)
            elif nn.Conv2d in types:
                i = types.index(nn.Conv2d)  # nn.Conv2d index
                if m[i].out_channels != nc:
                    m[i] = nn.Conv2d(m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None)

    def init_criterion(self):
        """Initialize the loss criterion for the ClassificationModel."""
        return v8ClassificationLoss()


class RTDETRDetectionModel(DetectionModel):
    """
    RTDETR (Real-time DEtection and Tracking using Transformers) Detection Model class.

    This class is responsible for constructing the RTDETR architecture, defining loss functions, and facilitating both
    the training and inference processes. RTDETR is an object detection and tracking model that extends from the
    DetectionModel base class.

    Attributes:
        cfg (str): The configuration file path or preset string. Default is 'rtdetr-l.yaml'.
        ch (int): Number of input channels. Default is 3 (RGB).
        nc (int, optional): Number of classes for object detection. Default is None.
        verbose (bool): Specifies if summary statistics are shown during initialization. Default is True.

    Methods:
        init_criterion: Initializes the criterion used for loss calculation.
        loss: Computes and returns the loss during training.
        predict: Performs a forward pass through the network and returns the output.
    """

    def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize the RTDETRDetectionModel.

        Args:
            cfg (str): Configuration file name or path.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes. Defaults to None.
            verbose (bool, optional): Print additional information during initialization. Defaults to True.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the RTDETRDetectionModel."""
        from ultralytics.models.utils.loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        """
        Compute the loss for the given batch of data.

        Args:
            batch (dict): Dictionary containing image and label data.
            preds (torch.Tensor, optional): Precomputed model predictions. Defaults to None.

        Returns:
            (tuple): A tuple containing the total loss and main three losses in a tensor.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        img = batch["img"]
        # NOTE: preprocess gt_bbox and gt_labels to list.
        bs = len(img)
        batch_idx = batch["batch_idx"]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
        }

        preds = self.predict(img, batch=targets) if preds is None else preds
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion(
            (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        )
        # NOTE: There are like 12 losses in RTDETR, backward with all losses but only show the main three losses.
        return sum(loss.values()), torch.as_tensor(
            [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
        )

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
            visualize (bool, optional): If True, save feature maps for visualization. Defaults to False.
            batch (dict, optional): Ground truth data for evaluation. Defaults to None.
            augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model[:-1]:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        head = self.model[-1]
        x = head([y[j] for j in head.f], batch)  # head inference
        return x


class WorldModel(DetectionModel):
    """YOLOv8 World Model."""

    def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 world model with given config and parameters."""
        self.txt_feats = torch.randn(1, nc or 80, 512)  # features placeholder
        self.clip_model = None  # CLIP model placeholder
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def set_classes(self, text, batch=80, cache_clip_model=True):
        """Set classes in advance so that model could do offline-inference without clip model."""
        try:
            import clip
        except ImportError:
            check_requirements("git+https://github.com/ultralytics/CLIP.git")
            import clip

        if (
            not getattr(self, "clip_model", None) and cache_clip_model
        ):  # for backwards compatibility of models lacking clip_model attribute
            self.clip_model = clip.load("ViT-B/32")[0]
        model = self.clip_model if cache_clip_model else clip.load("ViT-B/32")[0]
        device = next(model.parameters()).device
        text_token = clip.tokenize(text).to(device)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        self.txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
        self.model[-1].nc = len(text)

    def predict(self, x, profile=False, visualize=False, txt_feats=None, augment=False, embed=None):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
            visualize (bool, optional): If True, save feature maps for visualization. Defaults to False.
            txt_feats (torch.Tensor): The text features, use it if it's given. Defaults to None.
            augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(device=x.device, dtype=x.dtype)
        if len(txt_feats) != len(x):
            txt_feats = txt_feats.repeat(len(x), 1, 1)
        ori_txt_feats = txt_feats.clone()
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, C2fAttn):
                x = m(x, txt_feats)
            elif isinstance(m, WorldDetect):
                x = m(x, ori_txt_feats)
            elif isinstance(m, ImagePoolingAttn):
                txt_feats = m(x, txt_feats)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], txt_feats=batch["txt_feats"])
        return self.criterion(preds, batch)


class Ensemble(nn.ModuleList):
    """Ensemble of models."""

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Function generates the YOLO network's final layer."""
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C)
        return y, None  # inference, train output


# Functions ------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def temporary_modules(modules=None):
    """
    Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`).

    This function can be used to change the module paths during runtime. It's useful when refactoring code,
    where you've moved a module from one location to another, but you still want to support the old import
    paths for backwards compatibility.

    Args:
        modules (dict, optional): A dictionary mapping old module paths to new module paths.

    Example:
        ```python
        with temporary_modules({'old.module.path': 'new.module.path'}):
            import old.module.path  # this will now import new.module.path
        ```

    Note:
        The changes are only in effect inside the context manager and are undone once the context manager exits.
        Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger
        applications or libraries. Use this function with caution.
    """
    if not modules:
        modules = {}

    import importlib
    import sys

    try:
        # Set modules in sys.modules under their old name
        for old, new in modules.items():
            sys.modules[old] = importlib.import_module(new)

        yield
    finally:
        # Remove the temporary module paths
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


def torch_safe_load(weight):
    """
    This function attempts to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised,
    it catches the error, logs a warning message, and attempts to install the missing module via the
    check_requirements() function. After installation, the function again attempts to load the model using torch.load().

    Args:
        weight (str): The file path of the PyTorch model.

    Returns:
        (dict): The loaded PyTorch model.
    """
    from ultralytics.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix=".pt")
    file = attempt_download_asset(weight)  # search online if missing locally
    try:
        with temporary_modules(
            {
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
            }
        ):  # for legacy 8.0 Classify and Pose models
            ckpt = torch.load(file, map_location="cpu")

    except ModuleNotFoundError as e:  # e.name is missing module name
        if e.name == "models":
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained "
                    f"with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with "
                    f"YOLOv8 at https://github.com/ultralytics/ultralytics."
                    f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                    f"run a command with an official YOLOv8 model, i.e. 'yolo predict model=yolov8n.pt'"
                )
            ) from e
        LOGGER.warning(
            f"WARNING ⚠️ {weight} appears to require '{e.name}', which is not in ultralytics requirements."
            f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
            f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
            f"run a command with an official YOLOv8 model, i.e. 'yolo predict model=yolov8n.pt'"
        )
        check_requirements(e.name)  # install missing module
        ckpt = torch.load(file, map_location="cpu")

    if not isinstance(ckpt, dict):
        # File is likely a YOLO instance saved with i.e. torch.save(model, "saved_model.pt")
        LOGGER.warning(
            f"WARNING ⚠️ The file '{weight}' appears to be improperly saved or formatted. "
            f"For optimal results, use model.save('filename.pt') to correctly save YOLO models."
        )
        ckpt = {"model": ckpt.model}

    return ckpt, file  # load


def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    """Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a."""

    ensemble = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt, w = torch_safe_load(w)  # load ckpt
        args = {**DEFAULT_CFG_DICT, **ckpt["train_args"]} if "train_args" in ckpt else None  # combined args
        model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

        # Model compatibility updates
        model.args = args  # attach args to model
        model.pt_path = w  # attach *.pt file path to model
        model.task = guess_model_task(model)
        if not hasattr(model, "stride"):
            model.stride = torch.tensor([32.0])

        # Append
        ensemble.append(model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval())  # model in eval mode

    # Module updates
    for m in ensemble.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(ensemble) == 1:
        return ensemble[-1]

    # Return ensemble
    LOGGER.info(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[int(torch.argmax(torch.tensor([m.stride.max() for m in ensemble])))].stride
    assert all(ensemble[0].nc == m.nc for m in ensemble), f"Models differ in class counts {[m.nc for m in ensemble]}"
    return ensemble


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """Loads a single model weights."""
    ckpt, weight = torch_safe_load(weight)  # load ckpt
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  # combine model and default args, preferring model args
    model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

    # Model compatibility updates
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
    model.pt_path = weight  # attach *.pt file path to model
    model.task = guess_model_task(model)
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval()  # model in eval mode

    # Module updates
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt


def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")

    ch = [ch]

    tx=[3,256,256,512,512,1024,1024]

    ty=0
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        # m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # get module
        try:
            if m == 'node_mode':
                m = d[m]
                if len(args) > 0:
                    if args[0] == 'head_channel':
                        args[0] = int(d[args[0]])
            t = m
            m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]  # get module
        except:
            pass
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain

        if m in {
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            RepNCSPELAN4,
            ADown,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            MdC2f,
            CDC2f,
            C2f_Shufflenet,
            C2f_Invo,
            C2f_PKIModule,
            C2f_FEM,
            C2f_Faster,
            C2f_RG,
            A2C2f,
            C3k2,
            C2PSA,
        }:



            if f==-4:
                c1=3
                c2=args[0]
            else:
               c1, c2 = ch[f], args[0]

            # if f==-4:
            # #此时为下个backbonce,红外�?            #     c1=3
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)  # embed channels
                args[2] = int(
                    max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2]
                )  # num heads
            '''if m is C2f:
               if c1==768:
                  c1=1024
                  #print(c1)'''
            args = [c1, c2, *args[1:]]
            #print(len(args))
            if m in {BottleneckCSP, C1, C2, C2f, C2PSA, C2fAttn, C3, C3TR, C3Ghost, C3x, RepC3,C2f_PKIModule,C2f_FEM,C2f_PPA,C2f_RG,C2f_Faster,A2C2f,C3k2}:
                args.insert(2, n)  # number of repeats
                n = 1
                #print(len(args))

        elif m is FEM:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            tn=args[1]

            args=[c1,c2,tn]
        elif m is Fusion:
            args[0] = d[args[0]]
            c1, c2 = [ch[x] for x in f], (sum([ch[x] for x in f]) if args[0] == 'concat' else ch[f[0]])
            args = [c1, args[0]]
        elif m is ADD:
#            print("ch[f]", f, ch[f[0]])
            c2 = ch[f[0]]
            args = [c2]
        elif m is S2Attention:
            c1 = ch[f[0]]+ch[f[1]]
            c2 = ch[f[0]]
            args = [c1,c2]
        elif m is RIFusion:
            args = [args[0]]
        elif m in {SKAttention,GLF,NAM,GLCBAM,GCBAM,SACBAM,CSFM}:
            c1 = ch[f[0]]+ch[f[1]]
            c2 = ch[f[0]]
            args = [c1,c2]
        elif m is ShuffleAttention:
            c1 = ch[f[0]]+ch[f[1]]
            c2 = ch[f[0]]
            args = [c1,c2]
        elif m is SimAM:

            c1 = ch[f[0]]+ch[f[1]]
            c2 = ch[f[0]]
            args = [c1,c2]
        elif m is GAM_Attention:
            c1 = ch[f[0]]+ch[f[1]]
            c2 = ch[f[0]]
            args = [c1,c2]
        elif m is Concat2:

            c1 = ch[f[0]]+ch[f[1]]
            c2 = ch[f[0]]
            args = [c1,c2]
        elif m is Concat3:

            c1 = ch[f[0]]+ch[f[1]]
            c2 = ch[f[0]]
            args = [c1,c2]
        elif m is SEAttention:
            c1 = ch[f[0]]+ch[f[1]]
            c2 = ch[f[0]]
            args = [c1,c2]
        elif m is ECA:
            c1 = ch[f[0]]+ch[f[1]]
            c2 = ch[f[0]]
            args = [c1,c2]

        elif m is CBAM2:
            c1 = ch[f[0]]+ch[f[1]]
            c2 = ch[f[0]]
            args = [c1,c2]
        elif m is CoordAtt:
            c1 = ch[f[0]]+ch[f[1]]
            c2 = ch[f[0]]
            args = [c1,c2]
        elif m is AIFI:
            args = [ch[f], *args]
        elif m is returnlowir:
            c1,c2=ch[f],args[0]
            if c2 != nc:  # if not output
               c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1,c2]
        elif m is returnlowrgb:
            c1,c2=ch[f],args[0]
            if c2 != nc:  # if not output
               c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1,c2]
        elif m is returnfeature:
            c1, c2 = ch[f[0]], args[0]
            if c2 != nc:  # if not output
               c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1,c2]
        elif m in {HGStem, HGBlock}:
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1
        elif m is ResNetLayer:
            #print(c1)
            c2 = args[1] if args[3] else args[1] * 4
            #print(c2)
        elif m is ResNetLayer2:
            #print(c1)
            c2 = args[1] if args[3] else args[1] * 4
            #print(c2)
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in {Detect, AuxDetect, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn}:
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)

        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif m is frequencyfuse:
            c1, c2 = ch[f[0]], args[0]
            if c2 != nc:  # if not output
               c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
        elif m is Add:
            # print("ch[f]", f, ch[f[0]])
            c2 = ch[f[0]]
            args = [c2]
        elif m is Add2:
            # print("ch[f]", f, ch[f[0]])
            c2 = args[0]
            if c2 != nc:  # if not output
               c2 = make_divisible(min(c2, max_channels) * width, 8)
            # print("Add2 arg", args[0])
            args = [c2, args[1]]
        elif m is IRRGBGatedFusion:
            c1_source = ch[f] if isinstance(f, int) else ch[f[0]]
            c1, c2 = c1_source, args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            gate_bias = args[1] if len(args) > 1 else -2.0
            rgb_detail_filter = args[2] if len(args) > 2 else False
            wavelet_taylor = args[3] if len(args) > 3 else False
            positive_wavelet_taylor = args[4] if len(args) > 4 else False
            args = [c1, c2, gate_bias, rgb_detail_filter, wavelet_taylor, positive_wavelet_taylor]
        elif m in {TalorNet, AdapterTalorNet, P3AdapterTalorNet}:
            c1_source = ch[f] if isinstance(f, int) else ch[f[0]]
            c1, c2 = c1_source, args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if len(args) > 1 and isinstance(args[1], str):
                scale = args[1]
                taylor_order = args[2] if len(args) > 2 else 3
                use_vmamba = args[3] if len(args) > 3 else False
                quality_aware = args[4] if len(args) > 4 else False
            else:
                scale = "p2"
                taylor_order = args[1] if len(args) > 1 else 3
                use_vmamba = args[2] if len(args) > 2 else False
                quality_aware = args[3] if len(args) > 3 else False
            if m in {AdapterTalorNet, P3AdapterTalorNet}:
                adapter_drop_path = args[5] if len(args) > 5 else 0.0
                args = [c1, c2, scale, taylor_order, use_vmamba, quality_aware, adapter_drop_path]
            else:
                wavelet_taylor = args[5] if len(args) > 5 else False
                positive_wavelet_taylor = args[6] if len(args) > 6 else False
                args = [
                    c1,
                    c2,
                    scale,
                    taylor_order,
                    use_vmamba,
                    quality_aware,
                    wavelet_taylor,
                    positive_wavelet_taylor,
                ]
        elif m in {getreturntalorfuse, getreturntalorir}:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2]
        elif m is fusehd_second:
            c1, c2 = ch[f[0]], args[0]
            if c2 != nc:  # if not output
               c2 = make_divisible(min(c2, max_channels) * width, 8)
            # Keep scale parameter from yaml config (args[1])
            scale = args[1] if len(args) > 1 else 'p2'
            args = [c1, c2, scale]
        elif m is fusehd_second_optimized:
            c1, c2 = ch[f[0]], args[0]
            if c2 != nc:  # if not output
               c2 = make_divisible(min(c2, max_channels) * width, 8)
            # 支持多种写法：
            # [c2]
            # [c2, use_se, use_cross_group]
            # [c2, 'p1'/'p2'/'p3', use_se, use_cross_group, depth]
            scale = 'p2'
            use_se = True
            use_cross_group = True
            module_depth = 1

            if len(args) > 1:
                # 第二个参数若为字符串，则视为 scale
                if isinstance(args[1], str):
                    scale = args[1]
                    if len(args) > 2:
                        use_se = args[2]
                    if len(args) > 3:
                        use_cross_group = args[3]
                    if len(args) > 4:
                        module_depth = args[4]
                else:
                    # 否则按旧写法兼容：[c2, use_se, use_cross_group, depth]
                    use_se = args[1]
                    if len(args) > 2:
                        use_cross_group = args[2]
                    if len(args) > 3:
                        module_depth = args[3]

            # NOTE: `fusehd_second_optimized.__init__` 不接收 module_depth 参数。
            # 这里仅使用它来解析 YAML 里的可选配置，但不传入模块，避免参数个数不匹配报错。
            args = [c1, c2, scale, use_se, use_cross_group]
        elif m is fusehd_third_optimized:
            c1, c2 = ch[f[0]], args[0]
            if c2 != nc:  # if not output
               c2 = make_divisible(min(c2, max_channels) * width, 8)
            # Keep scale parameter from yaml config (args[1])
            scale = args[1] if len(args) > 1 else 'p2'
            # Optional: use_se and use_cross_group parameters (args[2] and args[3])
            use_se = args[2] if len(args) > 2 else True
            use_cross_group = args[3] if len(args) > 3 else True
            args = [c1, c2, scale, use_se, use_cross_group]
        elif m is fusehd_fourth:
            c1_source = ch[f] if isinstance(f, int) else ch[f[0]]
            c1, c2 = c1_source, args[0]
            if c2 != nc:  # if not output
               c2 = make_divisible(min(c2, max_channels) * width, 8)
            # Keep scale parameter from yaml config (args[1])
            scale = args[1] if len(args) > 1 else 'p2'
            # Optional: use_se and use_cross_group parameters (args[2] and args[3])
            use_se = args[2] if len(args) > 2 else True
            use_cross_group = args[3] if len(args) > 3 else True
            # Depth parameter for multi-level recursive loop (args[4])
            # Note: using 'module_depth' to avoid shadowing the global 'depth' scaling factor
            module_depth = args[4] if len(args) > 4 else 1
            args = [c1, c2, scale, use_se, use_cross_group, module_depth]
        elif m is spatial_fuse_ncssd:
            c1_source = ch[f] if isinstance(f, int) else ch[f[0]]
            c1, c2 = c1_source, args[0]
            if c2 != nc:  # if not output
               c2 = make_divisible(min(c2, max_channels) * width, 8)
            scale = args[1] if len(args) > 1 else 'p2'
            use_se = args[2] if len(args) > 2 else True
            use_cross_group = args[3] if len(args) > 3 else True
            module_depth = args[4] if len(args) > 4 else 1
            args = [c1, c2, scale, use_se, use_cross_group, module_depth]
        elif m is VMAMBA2BasicLayer:
            c1_source = ch[f] if isinstance(f, int) else ch[f[0]]
            c1, c2 = c1_source, args[0]
            if c2 != nc:  # if not output
               c2 = make_divisible(min(c2, max_channels) * width, 8)
            # Ensure c1 == c2 for VMAMBA2BasicLayer
            c2 = c1
            # Parse parameters: [channels, depth, drop_path, d_state, expand, headdim, ngroups, chunk_size, use_cpe]
            layer_depth = args[1] if len(args) > 1 else 1
            drop_path = args[2] if len(args) > 2 else 0.0
            d_state = args[3] if len(args) > 3 else 16
            expand = args[4] if len(args) > 4 else 2
            # Default headdim needs to divide d_inner=expand*c1 after width scaling
            headdim = args[5] if len(args) > 5 else make_divisible(min(c1, 64), 8)
            ngroups = args[6] if len(args) > 6 else 1
            chunk_size = args[7] if len(args) > 7 else 256
            use_cpe = args[8] if len(args) > 8 else True
            args = [c1, c2, layer_depth, drop_path, d_state, expand, headdim, ngroups, chunk_size, use_cpe]
        elif m is VSSLayer:
            c1_source = ch[f] if isinstance(f, int) else ch[f[0]]
            c1 = c2 = c1_source  # VSSLayer 直接使用前一层的输出通道数，不再应用 width scaling
            # Parse parameters: [dim, depth, attn_drop, drop_path, norm_layer, downsample, use_checkpoint, d_state]
            depth_layer = args[1] if len(args) > 1 else 1
            attn_drop = args[2] if len(args) > 2 else 0.0
            drop_path_rate = args[3] if len(args) > 3 else 0.0
            d_state = args[4] if len(args) > 4 else 16
            # 修复：使用实际的通道数 c1（已经被前一层 scaling 过），而不是 yaml 中的原始值 args[0]
            # 也不要再次应用 width scaling
            args = [c1, depth_layer, attn_drop, drop_path_rate, nn.LayerNorm, None, False, d_state]
        elif m is getreturnir:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if not output
               c2 = make_divisible(min(c2, max_channels) * width, 8)
            # print("Add2 arg", args[0])
            args = [c1, c2]
        elif m is getreturnfuse:
            c1, c2 = 2*ch[f], args[0]
            if c2 != nc:  # if not output
               c2 = make_divisible(min(c2, max_channels) * width, 8)
            #print("Add2 arg", args[0])
            args = [c1, c2]
        elif m is getreturnrgb:
           c1, c2 = ch[f], args[0]
           if c2 != nc:  # if not output
               c2 = make_divisible(min(c2, max_channels) * width, 8)
            # print("Add2 arg", args[0])
           args = [c1, c2]
        elif m is getreturnfeature:
           c1, c2 = ch[f], args[0]
           if c2 != nc:  # if not output
               c2 = make_divisible(min(c2, max_channels) * width, 8)
            # print("Add2 arg", args[0])
           args = [c1, c2]
        elif m is getreturnfeature2:
           c1, c2 = ch[f[0]], args[0]
           if c2 != nc:  # if not output
               c2 = make_divisible(min(c2, max_channels) * width, 8)
            # print("Add2 arg", args[0])
           args = [c1, c2]
        elif m is getreturnfhigh:
            c1=256
            c2=args[0]
            if c2 != nc:  # if not output
               c2 = make_divisible(min(c2, max_channels) * width, 8)
            # print("Add2 arg", args[0])
            args = [c1, c2]
        else:
            c2 = ch[f]
        #print(m,len(args))



        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module



        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}")  # print

        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x not in [-1,-3,-4])  # append to savelist


        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)

    return nn.Sequential(*layers), sorted(save)


def yaml_model_load(path):
    """Load a YOLOv8 model from a YAML file."""
    import re

    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(f"WARNING ⚠️ Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.")
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)  # model dict
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d


def guess_model_scale(model_path):
    """
    Takes a path to a YOLO model's YAML file as input and extracts the size character of the model's scale. The function
    uses regular expression matching to find the pattern of the model scale in the YAML file name, which is denoted by
    n, s, m, l, or x. The function returns the size character of the model scale as a string.

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale, which can be n, s, m, l, or x.
    """
    with contextlib.suppress(AttributeError):
        import re

        return re.search(r"yolov\d+([nslmx])", Path(model_path).stem).group(1)  # n, s, m, l, or x
    return ""


def guess_model_task(model):
    """
    Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (nn.Module | dict): PyTorch model or model configuration in YAML format.

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose').

    Raises:
        SyntaxError: If the task of the model could not be determined.
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg["head"][-1][-2].lower()  # output module name
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"
        if m == "detect":
            return "detect"
        if m == "segment":
            return "segment"
        if m == "pose":
            return "pose"
        if m == "obb":
            return "obb"

    # Guess from model cfg
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)

    # Guess from PyTorch model
    if isinstance(model, nn.Module):  # PyTorch model
        for x in "model.args", "model.model.args", "model.model.model.args":
            with contextlib.suppress(Exception):
                return eval(x)["task"]
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))

        for m in model.modules():
            if isinstance(m, Segment):
                return "segment"
            elif isinstance(m, Classify):
                return "classify"
            elif isinstance(m, Pose):
                return "pose"
            elif isinstance(m, OBB):
                return "obb"
            elif isinstance(m, (Detect, WorldDetect)):
                return "detect"

    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        if "-seg" in model.stem or "segment" in model.parts:
            return "segment"
        elif "-cls" in model.stem or "classify" in model.parts:
            return "classify"
        elif "-pose" in model.stem or "pose" in model.parts:
            return "pose"
        elif "-obb" in model.stem or "obb" in model.parts:
            return "obb"
        elif "detect" in model.parts:
            return "detect"

    # Unable to determine task from model
    LOGGER.warning(
        "WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. "
        "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'."
    )
    return "detect"  # assume detect
