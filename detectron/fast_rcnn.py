# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
import scipy.stats


from .layers.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple
from .layers.box_regression import Box2BoxTransform, _dense_box_regression_loss
from .layers.structures import Boxes, Instances


def fast_rcnn_inference(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    ncc_scores: List[torch.Tensor],
    gt_classes: torch.Tensor,
    box_features: torch.Tensor,
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, None, image_shape, score_thresh, nms_thresh, topk_per_image
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ] if ncc_scores is None else [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, ncc_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
        )
        for scores_per_image, boxes_per_image, ncc_per_image, image_shape in zip(scores, boxes, ncc_scores, image_shapes)
    ]
    features = None
    if gt_classes is not None:
        assert len(result_per_image) == gt_classes.shape[0]
        features = {"pos_features":[],"neg_features":[]}
        pos_mask = gt_classes == 0
        neg_mask = gt_classes == 1
        for b in range(len(result_per_image)):
            pos_features = box_features[b][pos_mask[b]]
            neg_features = box_features[b][neg_mask[b]]
            features["pos_features"].append(pos_features)
            features["neg_features"].append(neg_features)
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image], features



def fast_rcnn_inference_single_image(
    boxes,
    scores,
    ncc_score,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    ncc_score = ncc_score[filter_mask][keep] if ncc_score is not None else None
    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    if ncc_score is not None:
        result.ncc_scores = ncc_score
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]


class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:

    1. proposal-to-detection box regression deltas
    2. classification scores
    """


    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        box2box_transform,
        num_classes: int,
        test_score_thresh: float = 0.0,
        test_nms_thresh: float = 0.5,
        test_topk_per_image: int = 100,
        cls_agnostic_bbox_reg: bool = False,
        smooth_l1_beta: float = 0.0,
        box_reg_loss_type: str = "smooth_l1",
        loss_weight: Union[float, Dict[str, float]] = 1.0,
        use_fed_loss: bool = False,
        use_sigmoid_ce: bool = False,
        get_fed_loss_cls_weights: Optional[Callable] = None,
        fed_loss_num_classes: int = 50,
        pu_learning: bool = False,
        pi: float = 0.0,
        num_particle: int = -1,
        radius: int = 3,
        total_regions: int = -1,
        beta: float = 0.0,
        use_ncc_scores: bool = False,
        contrastive_train: bool = False,
        cl_mlp_in_dim: int = 50176,
        cl_mlp_out_dim: int = 256,
        cl_loss_lambda: float = 0.1,
        cl_pos_num: int = 15,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou",
                "diou", "ciou"
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all losses, or a dict of individual weightings. Valid dict keys are:
                    * "loss_cls": applied to classification loss
                    * "loss_box_reg": applied to box regression loss
            use_fed_loss (bool): whether to use federated loss which samples additional negative
                classes to calculate the loss
            use_sigmoid_ce (bool): whether to calculate the loss using weighted average of binary
                cross entropy with logits. This could be used together with federated loss
            get_fed_loss_cls_weights (Callable): a callable which takes dataset name and frequency
                weight power, and returns the probabilities to sample negative classes for
                federated loss. The implementation can be found in
                detectron2/data/detection_utils.py
            fed_loss_num_classes (int): number of federated classes to keep in total
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        self.num_classes = num_classes
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        # prediction layer for num_classes foreground classes and one background class (hence + 1)
        self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        if isinstance(loss_weight, float):
            loss_weight = {"loss_cls": loss_weight, "loss_box_reg": loss_weight}
        self.loss_weight = loss_weight
        self.use_fed_loss = use_fed_loss
        self.use_sigmoid_ce = use_sigmoid_ce
        self.fed_loss_num_classes = fed_loss_num_classes

        self.pu_learning = pu_learning
        self.pi = pi
        self.beta = beta
        self.use_ncc_scores = use_ncc_scores
        self.contrastive_train = contrastive_train
        if num_particle > 0:
            grid = np.linspace(-radius, radius, 2*radius+1)
            xx = np.zeros((2*radius+1, 2*radius+1)) + grid[:,np.newaxis]
            yy = np.zeros((2*radius+1, 2*radius+1)) + grid[np.newaxis]
            d2 = xx**2 + yy**2
            mask = (d2 <= radius**2).astype(int)
            pixels_per_particle = mask.sum()
            self.pi = pixels_per_particle*num_particle/total_regions

        if self.use_ncc_scores:
            self.ncc_preds = nn.Linear(input_size, 1)
            loss_weight["loss_ncc"] = loss_weight["loss_cls"]

        if self.contrastive_train:
            self.cl_mlp = nn.Linear(cl_mlp_in_dim,cl_mlp_out_dim)
            loss_weight["loss_cl"] = loss_weight["loss_cls"] * cl_loss_lambda
            self.positive_num = cl_pos_num

        if self.use_fed_loss:
            assert self.use_sigmoid_ce, "Please use sigmoid cross entropy loss with federated loss"
            fed_loss_cls_weights = get_fed_loss_cls_weights()
            assert (
                len(fed_loss_cls_weights) == self.num_classes
            ), "Please check the provided fed_loss_cls_weights. Their size should match num_classes"
            self.register_buffer("fed_loss_cls_weights", fed_loss_cls_weights)

    def forward(self, x):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        if self.use_ncc_scores:
            ncc_scores = self.ncc_preds(x)
            return scores, proposal_deltas, ncc_scores
        return scores, proposal_deltas

    def losses(self, predictions, proposals, box_features):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        if self.use_ncc_scores:
            scores, proposal_deltas, ncc_scores = predictions
        else:
            scores, proposal_deltas = predictions

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        # _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
            if self.use_ncc_scores:
                if all([p.has("ncc_scores") for p in proposals]):
                    gt_ncc_scores = torch.unsqueeze(cat([p.ncc_scores for p in proposals], dim=0), dim=-1)
                else:
                    gt_ncc_scores = None
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        if self.use_sigmoid_ce:
            loss_cls = self.sigmoid_cross_entropy_loss(scores, gt_classes)
        else:
            if self.pu_learning:
                pos_mask = gt_classes < self.num_classes
                neg_mask = gt_classes == self.num_classes
                # GE_binomial in topaz
                # cls_loss = cross_entropy(scores[pos_mask], gt_classes[pos_mask], reduction="mean")
                # N = neg_mask.sum().item()
                # p_hat = torch.sigmoid(scores[neg_mask])
                # q_mu = p_hat.sum()
                # q_var = torch.sum(p_hat*(1-p_hat))

                # count_vector = torch.arange(0,N+1).float()
                # count_vector = count_vector.to(q_mu.device)

                # q_discrete = -0.5*(q_mu-count_vector)**2/(q_var + 1e-10) # add small epsilon to prevent NaN
                # q_discrete = F.softmax(q_discrete, dim=0)

                # ## KL of w from the binomial distribution with pi
                # log_binom = scipy.stats.binom.logpmf(np.arange(0,N+1),N,self.pi)
                # log_binom = torch.from_numpy(log_binom).float()
                # if q_var.is_cuda:
                #     log_binom = log_binom.cuda()
                # log_binom = Variable(log_binom)

                # ge_penalty = -torch.sum(log_binom*q_discrete)
                # # q_entropy = 0.5*(torch.log(q_var) + np.log(2*np.pi) + 1)
                # # ge_penalty = ge_penalty + q_entropy

                # loss_cls = cls_loss + ge_penalty
                # PU in topaz
                loss_cls_pp = cross_entropy(
                    scores[pos_mask],
                    gt_classes[pos_mask],
                    reduction="mean",
                )
                loss_cls_pn = cross_entropy(
                    scores[pos_mask],
                    0 * gt_classes[pos_mask],
                    reduction="mean",
                ) # estimate loss for calling positives negative
                loss_cls_un =cross_entropy(
                    scores[neg_mask],
                    gt_classes[neg_mask],
                    reduction="mean",
                )
                loss_cls_u = loss_cls_un - loss_cls_pn * self.pi # estimate loss for negative data in unlabeled set
                if loss_cls_u.item() < -self.beta:
                    loss_cls = loss_cls_pp * self.pi
                else:
                    loss_cls = loss_cls_pp * self.pi + loss_cls_u
            else:
                loss_cls = cross_entropy(scores, gt_classes, reduction="mean")

        losses = {
            "loss_cls": loss_cls,
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }
        if self.use_ncc_scores and gt_ncc_scores is not None:
            pos_mask = torch.unsqueeze(gt_classes < self.num_classes, dim=-1)
            losses["loss_ncc"] = F.mse_loss((ncc_scores * (gt_ncc_scores != 0))[pos_mask], (gt_ncc_scores * (gt_ncc_scores != 0))[pos_mask])
            # losses["loss_ncc"] = F.mse_loss(ncc_scores[pos_mask], gt_ncc_scores[pos_mask])
        if self.contrastive_train:
            losses["loss_cl"] = self.contrastive_loss(proposals, box_features, gt_classes, self.positive_num)
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    # Implementation from https://github.com/xingyizhou/CenterNet2/blob/master/projects/CenterNet2/centernet/modeling/roi_heads/fed_loss.py  # noqa
    # with slight modifications
    def get_fed_loss_classes(self, gt_classes, num_fed_loss_classes, num_classes, weight):
        """
        Args:
            gt_classes: a long tensor of shape R that contains the gt class label of each proposal.
            num_fed_loss_classes: minimum number of classes to keep when calculating federated loss.
            Will sample negative classes if number of unique gt_classes is smaller than this value.
            num_classes: number of foreground classes
            weight: probabilities used to sample negative classes

        Returns:
            Tensor:
                classes to keep when calculating the federated loss, including both unique gt
                classes and sampled negative classes.
        """
        unique_gt_classes = torch.unique(gt_classes)
        prob = unique_gt_classes.new_ones(num_classes + 1).float()
        prob[-1] = 0
        if len(unique_gt_classes) < num_fed_loss_classes:
            prob[:num_classes] = weight.float().clone()
            prob[unique_gt_classes] = 0
            sampled_negative_classes = torch.multinomial(
                prob, num_fed_loss_classes - len(unique_gt_classes), replacement=False
            )
            fed_loss_classes = torch.cat([unique_gt_classes, sampled_negative_classes])
        else:
            fed_loss_classes = unique_gt_classes
        return fed_loss_classes

    # Implementation from https://github.com/xingyizhou/CenterNet2/blob/master/projects/CenterNet2/centernet/modeling/roi_heads/custom_fast_rcnn.py#L113  # noqa
    # with slight modifications
    def sigmoid_cross_entropy_loss(self, pred_class_logits, gt_classes):
        """
        Args:
            pred_class_logits: shape (N, K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class
            gt_classes: a long tensor of shape R that contains the gt class label of each proposal.
        """
        if pred_class_logits.numel() == 0:
            return pred_class_logits.new_zeros([1])[0]

        N = pred_class_logits.shape[0]
        K = pred_class_logits.shape[1] - 1

        target = pred_class_logits.new_zeros(N, K + 1)
        target[range(len(gt_classes)), gt_classes] = 1
        target = target[:, :K]

        cls_loss = F.binary_cross_entropy_with_logits(
            pred_class_logits[:, :-1], target, reduction="none"
        )

        if self.use_fed_loss:
            fed_loss_classes = self.get_fed_loss_classes(
                gt_classes,
                num_fed_loss_classes=self.fed_loss_num_classes,
                num_classes=K,
                weight=self.fed_loss_cls_weights,
            )
            fed_loss_classes_mask = fed_loss_classes.new_zeros(K + 1)
            fed_loss_classes_mask[fed_loss_classes] = 1
            fed_loss_classes_mask = fed_loss_classes_mask[:K]
            weight = fed_loss_classes_mask.view(1, K).expand(N, K).float()
        else:
            weight = 1

        loss = torch.sum(cls_loss * weight) / N
        return loss

    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes):
        """
        Args:
            proposal_boxes/gt_boxes are tensors with the same shape (R, 4 or 5).
            pred_deltas has shape (R, 4 or 5), or (R, num_classes * (4 or 5)).
            gt_classes is a long tensor of shape R, the gt class label of each proposal.
            R shall be the number of proposals.
        """
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        # Regression loss is only computed for foreground proposals (those matched to a GT)
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]

        loss_box_reg = _dense_box_regression_loss(
            [proposal_boxes[fg_inds]],
            self.box2box_transform,
            [fg_pred_deltas.unsqueeze(0)],
            [gt_boxes[fg_inds]],
            ...,
            self.box_reg_loss_type,
            self.smooth_l1_beta,
        )

        # The reg loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        return loss_box_reg / max(gt_classes.numel(), 1.0)  # return 0 if empty

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances], box_features):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        ncc_scores = None
        if self.use_ncc_scores:
            ncc_scores = self.predict_ncc(predictions, proposals)
        if proposals[0].has("gt_classes") and self.contrastive_train:
            gt_classes = (
                cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
            )
            B = len(proposals)
            gt_classes = gt_classes.reshape(B, gt_classes.shape[0] // B)
            box_features = F.normalize(self.cl_mlp(box_features.reshape(gt_classes.shape[0], gt_classes.shape[1], -1)), p=2, dim=-1)
        else:
            gt_classes = None
            box_features = None
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            ncc_scores,
            gt_classes,
            box_features,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

    def contrastive_loss(self, proposals, features, gt_classes, tau=0.07, positive_num=15):
        losses = []

        start_idx = 0
        for proposal in proposals:
            num_proposals = len(proposal)
            end_idx = start_idx + num_proposals

            instance_features = features[start_idx:end_idx]
            pos_mask = gt_classes[start_idx:end_idx] == 1
            neg_mask = gt_classes[start_idx:end_idx] == 0

            pos_features = instance_features[pos_mask].flatten(start_dim=1)
            neg_features = instance_features[neg_mask].flatten(start_dim=1)

            start_idx = end_idx

            if len(pos_features) > 0 and len(neg_features) > 0:
                pos_features = F.normalize(self.cl_mlp(pos_features[:positive_num]), p=2, dim=1)
                neg_features = F.normalize(self.cl_mlp(neg_features), p=2, dim=1)

                pp_sim = pos_features @ pos_features.T #[S1, S1]
                pn_sim = pos_features @ neg_features.T #[S1, S2]
                S1 = pp_sim.shape[1]
                S2 = pn_sim.shape[1]
                pp_sim = pp_sim.unsqueeze(-1)
                pn_sim = pn_sim.unsqueeze(0).expand(S1, S1, S2)
                # loss = max(1 + self.alpha * cross_sim.mean() - (1 - self.alpha) * self_sim.mean(),0)
                predictions = torch.cat([pp_sim, pn_sim], dim=-1) # [S1, S1, 1+S2]
                predictions = predictions.flatten(0, 1) / tau
                targets = torch.zeros(S1 * S1, dtype=torch.long, device=predictions.device)
                # targets = torch.cat([torch.ones(S1, dtype=torch.long), torch.zeros(S2, dtype=torch.long)])
                loss = F.cross_entropy(predictions, targets)

                losses.append(loss)

        return torch.stack(losses).mean() if losses else torch.tensor(0.0).to(features.device)

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_classes`` are expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        scores, proposal_deltas = predictions
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[
                torch.arange(N, dtype=torch.long, device=predict_boxes.device), gt_classes
            ]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        proposal_deltas = predictions[1]
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes,
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        scores = predictions[0]
        num_inst_per_image = [len(p) for p in proposals]
        if self.use_sigmoid_ce:
            probs = scores.sigmoid()
        else:
            probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)

    def predict_ncc(self, predictions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], proposals: List[Instances]):
        ncc_scores = predictions[2]
        num_inst_per_image = [len(p) for p in proposals]
        return ncc_scores.split(num_inst_per_image, dim=0)
