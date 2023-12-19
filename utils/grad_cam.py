import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.image import show_cam_on_image
from ultralytics.nn.tasks import DetectionModel as Model
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.torch_utils import intersect_dicts


class yolov8_heatmap:
    def __init__(self, weight, cfg, device, layer, backward_type):
        self.backward_type = backward_type

        device = torch.device(device)
        ckpt = torch.load(weight)
        model_names = ckpt["model"].names
        csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
        model = Model(cfg, ch=3, nc=len(model_names)).to(device)
        csd = intersect_dicts(csd, model.state_dict(), exclude=["anchor"])  # intersect
        model.load_state_dict(csd, strict=False)  # load
        model.eval()

        self.target_layers = [eval(layer)]

        self.__dict__.update(locals())

    def post_process(self, result):
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
        return (
            torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]],
            torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]],
            xywh2xyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]])
            .cpu()
            .detach()
            .numpy(),
        )

    def __call__(self, img_path=None, img=None):
        # img process
        if img is None:
            img = cv2.imread(img_path)
        img = cv2.resize(img, (640, 640))
        img = np.float32(img) / 255.0
        tensor = (
            torch.from_numpy(np.transpose(img, axes=[2, 0, 1]))
            .unsqueeze(0)
            .to(self.device)
        )

        # init ActivationsAndGradients
        grads = ActivationsAndGradients(
            self.model, self.target_layers, reshape_transform=None
        )

        # get ActivationsAndResult
        result = grads(tensor)
        activations = grads.activations[0].cpu().detach().numpy()

        # postprocess to yolo output
        post_result, pre_post_boxes, post_boxes = self.post_process(result[0])

        self.model.zero_grad()

        if self.backward_type == "class" or self.backward_type == "all":
            score = post_result[0].max()
            score.backward(retain_graph=True)

        if self.backward_type == "box" or self.backward_type == "all":
            for j in range(4):
                score = pre_post_boxes[0, j]
                score.backward(retain_graph=True)

        # process heatmap
        if self.backward_type == "class":
            gradients = grads.gradients[0]
        elif self.backward_type == "box":
            gradients = (
                grads.gradients[0]
                + grads.gradients[1]
                + grads.gradients[2]
                + grads.gradients[3]
            )
        else:
            gradients = (
                grads.gradients[0]
                + grads.gradients[1]
                + grads.gradients[2]
                + grads.gradients[3]
                + grads.gradients[4]
            )

        b, k, u, v = gradients.size()
        weights = GradCAM.get_cam_weights(
            GradCAM, None, None, None, activations, gradients.detach().numpy()
        )
        weights = weights.reshape((b, k, 1, 1))
        saliency_map = np.sum(weights * activations, axis=1)
        saliency_map = np.squeeze(np.maximum(saliency_map, 0))
        saliency_map = cv2.resize(saliency_map, (tensor.size(3), tensor.size(2)))
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min) / (
            saliency_map_max - saliency_map_min
        )

        # add heatmap to image
        return show_cam_on_image(img.copy(), saliency_map, use_rgb=True)
