from transformers.models.detr.modeling_detr import (
    DetrObjectDetectionOutput, build_position_encoding, DetrDecoder,
    DetrMLPPredictionHead, DetrHungarianMatcher, DetrLoss
)
from transformers import DetrConfig
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from einops import rearrange


class DetrDecoderForObjectDetection(nn.Module):
    def __init__(self, config: DetrConfig, d_model=10, d_hidden=100):
        super().__init__()
        self.config = config
        self.object_queries = build_position_encoding(config)
        self.query_position_embeddings = nn.Embedding(config.num_queries, d_model)
        self.decoder = DetrDecoder(config)

        self.bbox_predictor = DetrMLPPredictionHead(
            input_dim=d_model, hidden_dim=100, output_dim=4, num_layers=3
        )

        self.class_labels_classifier = nn.Linear(
            config.d_model, config.num_labels + 1
        )

    # taken from https://github.com/facebookresearch/detr/blob/master/models/detr.py
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[List[dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        loss_only: Optional[bool] = None,
        pred_boxes: Optional[torch.FloatTensor] = None,
        logits: Optional[torch.FloatTensor] = None
    ) -> Union[Tuple[torch.FloatTensor], DetrObjectDetectionOutput]:

        if not loss_only:
            #return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            pixel_values = inputs_embeds.permute(0, 3, 1, 2)
            batch_size, num_channels, height, width = pixel_values.shape
            pixel_mask = torch.ones(((batch_size, height, width))).to(pixel_values.device)

            object_queries = self.object_queries(pixel_values, pixel_mask)

            object_queries = object_queries.flatten(2).permute(0, 2, 1)

            query_position_embeddings = self.query_position_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)
            queries = torch.zeros_like(query_position_embeddings)

            inputs_embeds = rearrange(inputs_embeds, 'b h w c -> b (h w) c')

            decoder_outputs = self.decoder(
                inputs_embeds=queries,
                attention_mask=None,
                object_queries=object_queries,
                query_position_embeddings=query_position_embeddings,
                encoder_hidden_states=inputs_embeds,
            )

            sequence_output = decoder_outputs[0]

            # class logits + predicted bounding boxes
            logits = self.class_labels_classifier(sequence_output)
            pred_boxes = self.bbox_predictor(sequence_output).sigmoid()

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            # First: create the matcher
            matcher = DetrHungarianMatcher(
                class_cost=self.config.class_cost, bbox_cost=self.config.bbox_cost, giou_cost=self.config.giou_cost
            )
            # Second: create the criterion
            losses = ["labels", "boxes", "cardinality"]
            criterion = DetrLoss(
                matcher=matcher,
                num_classes=self.config.num_labels,
                eos_coef=self.config.eos_coefficient,
                losses=losses,
            )
            criterion.to(pred_boxes.device)
            # Third: compute the losses, based on outputs and labels
            outputs_loss = {}
            outputs_loss["logits"] = logits
            outputs_loss["pred_boxes"] = pred_boxes

            if self.config.auxiliary_loss:
                # broken?
                raise NotImplementedError
                # intermediate = outputs.intermediate_hidden_states if return_dict else outputs[4]
                # outputs_class = self.class_labels_classifier(intermediate)
                # outputs_coord = self.bbox_predictor(intermediate).sigmoid()
                # auxiliary_outputs = self._set_aux_loss(outputs_class, outputs_coord)
                # outputs_loss["auxiliary_outputs"] = auxiliary_outputs

            loss_dict = criterion(outputs_loss, labels)
            # Fourth: compute total loss, as a weighted sum of the various losses
            weight_dict = {"loss_ce": 1, "loss_bbox": self.config.bbox_loss_coefficient}
            weight_dict["loss_giou"] = self.config.giou_loss_coefficient
            if self.config.auxiliary_loss:
                aux_weight_dict = {}
                for i in range(self.config.decoder_layers - 1):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        if not return_dict:
            if auxiliary_outputs is not None:
                output = (pred_boxes) + auxiliary_outputs
            else:
                output = (pred_boxes, logits)
            return ((loss, loss_dict)) + output if loss is not None else output

        return DetrObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
        )


def _test():
    import torch
    import kwimage
    num = 10
    true_boxes = kwimage.Boxes.random(num).tensor()
    print(true_boxes.tensor)
    inputs = torch.rand(num, 14, 14, 256)
    config = DetrConfig()
    regress = DetrDecoderForObjectDetection(config, 256)
    energy = regress(inputs)
    energy.retain_grad()
    outputs = energy.sigmoid()
    outputs.retain_grad()
    out_boxes = kwimage.Boxes(outputs, 'cxywh')
    ious = out_boxes.ious(true_boxes)
    loss = ious.sum()
    print(loss)
    loss.backward()


if __name__ == '__main__':
    _test()
