#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
# Modified from bevformer (https://github.com/fundamentalvision/BEVFormer)        #
#---------------------------------------------------------------------------------#

import copy
import jittor
import jittor.nn as jnn

from jtmmcv.models.bricks import Linear
from jtmmcv.models.utils import bias_init_with_prob
from jtmmcv.utils import digit_version

from jtmmcv.core.utils import (multi_apply, reduce_mean)
from jtmmcv.models.utils.transformer import inverse_sigmoid
from jtmmcv.models import HEADS
from jtmmcv.models.dense_heads import DETRHead
from jtmmcv.core.bbox.coder import build_bbox_coder
from jtmmcv.core.bbox.util import normalize_bbox
from jtmmcv.utils import force_fp32, auto_fp16
from jtmmcv.utils.general import nan_to_num


@HEADS.register_module()
class BEVFormerTrackHead(DETRHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 past_steps=4,
                 fut_steps=4,
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        self.with_box_refine = with_box_refine

        assert as_two_stage is False, 'as_two_stage is not supported yet.'
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        self.past_steps = past_steps
        self.fut_steps = fut_steps
        super(BEVFormerTrackHead, self).__init__(
            *args, transformer=transformer, **kwargs)
        self.code_weights = jnn.Parameter(jittor.array(
            self.code_weights).stop_grad())

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(jnn.LayerNorm(self.embed_dims))
            cls_branch.append(jnn.ReLU())
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = jnn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(jnn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = jnn.Sequential(*reg_branch)

        past_traj_reg_branch = []
        for _ in range(self.num_reg_fcs):
            past_traj_reg_branch.append(
                Linear(self.embed_dims, self.embed_dims))
            past_traj_reg_branch.append(jnn.ReLU())
        past_traj_reg_branch.append(
            Linear(self.embed_dims, (self.past_steps + self.fut_steps)*2))
        past_traj_reg_branch = jnn.Sequential(*past_traj_reg_branch)


        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = jnn.ModuleList()
            for _ in range(num_pred):
                layers = []
                for _ in range(self.num_reg_fcs):  # possible bug here (num_cls_fcs)
                    layers.append(Linear(self.embed_dims, self.embed_dims))
                    layers.append(jnn.LayerNorm(self.embed_dims))
                    layers.append(jnn.ReLU())
                layers.append(Linear(self.embed_dims, self.cls_out_channels))
                self.cls_branches.append(jnn.Sequential(*layers))
                
            self.reg_branches = jnn.ModuleList()
            for _ in range(num_pred):
                layers = []
                for _ in range(self.num_reg_fcs):
                    layers.append(Linear(self.embed_dims, self.embed_dims))
                    layers.append(jnn.ReLU())
                layers.append(Linear(self.embed_dims, self.code_size))
                self.reg_branches.append(jnn.Sequential(*layers))
                
            self.past_traj_reg_branches = jnn.ModuleList()
            for _ in range(num_pred):
                layers = []
                for _ in range(self.num_reg_fcs):
                    layers.append(Linear(self.embed_dims, self.embed_dims))
                    layers.append(jnn.ReLU())
                layers.append(Linear(self.embed_dims, (self.past_steps + self.fut_steps)*2))
                self.past_traj_reg_branches.append(jnn.Sequential(*layers))
                    
        else:
            self.cls_branches = jnn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = jnn.ModuleList(
                [reg_branch for _ in range(num_pred)])
            self.past_traj_reg_branches = jnn.ModuleList(
                [past_traj_reg_branch for _ in range(num_pred)])
        if not self.as_two_stage:
            self.bev_embedding = jnn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                jittor.init.constant_(m[-1].bias, bias_init)
    
    def get_bev_features(self, mlvl_feats, img_metas, prev_bev=None):
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = jittor.zeros((bs, self.bev_h, self.bev_w)).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        bev_embed = self.transformer.get_bev_features(
            mlvl_feats,
            bev_queries,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h,
                         self.real_w / self.bev_w),
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            img_metas=img_metas,
        )
        return bev_embed, bev_pos

    def get_detections(
        self, 
        bev_embed,
        object_query_embeds=None,
        ref_points=None,
        img_metas=None,
    ):
        assert bev_embed.shape[0] == self.bev_h * self.bev_w
        hs, init_reference, inter_references = self.transformer.get_states_and_refs(
            bev_embed,
            object_query_embeds,
            self.bev_h,
            self.bev_w,
            reference_points=ref_points,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            cls_branches=self.cls_branches if self.as_two_stage else None,
            img_metas=img_metas,
        )
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        outputs_trajs = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                # reference = init_reference
                reference = ref_points.sigmoid()
            else:
                reference = inter_references[lvl - 1]
                # ref_size_base = inter_box_sizes[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])  # xydxdyxdz
            outputs_past_traj = self.past_traj_reg_branches[lvl](hs[lvl]).view(
                tmp.shape[0], -1, self.past_steps + self.fut_steps, 2)
            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

            last_ref_points = jittor.concat(
                [tmp[..., 0:2], tmp[..., 4:5]], dim=-1,
            )

            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                             self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                             self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                             self.pc_range[2]) + self.pc_range[2])

            # tmp[..., 2:4] = tmp[..., 2:4] + ref_size_basse[..., 0:2]
            # tmp[..., 5:6] = tmp[..., 5:6] + ref_size_basse[..., 2:3]

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_trajs.append(outputs_past_traj)
        outputs_classes = jittor.stack(outputs_classes)
        outputs_coords = jittor.stack(outputs_coords)
        outputs_trajs = jittor.stack(outputs_trajs)
        last_ref_points = inverse_sigmoid(last_ref_points)
        outs = {
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'all_past_traj_preds': outputs_trajs,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
            'last_ref_points': last_ref_points,
            'query_feats': hs,
        }
        return outs
        
    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (jittor.Var): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (jittor.Var): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (jittor.Var): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (jittor.Var): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (jittor.Var, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[jittor.Var]: a tuple containing the following for one image.
                - labels (jittor.Var): Labels of each image.
                - label_weights (jittor.Var]): Label weights of each image.
                - bbox_targets (jittor.Var): BBox targets of each image.
                - bbox_weights (jittor.Var): BBox weights of each image.
                - pos_inds (jittor.Var): Sampled positive indices for each image.
                - neg_inds (jittor.Var): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=jittor.int64)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = jittor.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = jittor.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (jittor.Var): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (jittor.Var): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[jittor.Var]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[jittor.Var]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[jittor.Var], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, jittor.Var]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = jittor.concat(labels_list, 0)
        label_weights = jittor.concat(label_weights_list, 0)
        bbox_targets = jittor.concat(bbox_targets_list, 0)
        bbox_weights = jittor.concat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new(cls_avg_factor))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new(num_total_pos)
        num_total_pos = jittor.clamp(reduce_mean(num_total_pos), min_v=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = jittor.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan,
                                                               :10], bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)
        loss_cls = nan_to_num(loss_cls)
        loss_bbox = nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[jittor.Var]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[jittor.Var]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (jittor.Var): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (jittor.Var): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-jittor.Var with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (jittor.Var): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (jittor.Var): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[jittor.Var], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, jittor.Var]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        num_dec_layers = len(all_cls_scores)


        gt_bboxes_list = [jittor.concat(
            (gt_bboxes.gravity_center, gt_bboxes[:, 3:]),
            dim=1) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                jittor.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']

            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            code_size = bboxes.shape[-1]
            bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']
            bbox_index = preds['bbox_index']
            mask = preds['mask']

            ret_list.append([bboxes, scores, labels, bbox_index, mask])

        return ret_list
