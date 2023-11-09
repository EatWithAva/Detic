import sys
import cv2
import tempfile
from cog import BasePredictor, Input, Path
import time
import torch

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Detic libraries
sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from detic.modeling.text.text_encoder import build_text_encoder

from typing import Any

class Predictor(BasePredictor):
    def setup(self):
        cfg = get_cfg()
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file("configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
        cfg.MODEL.WEIGHTS = 'models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
        # cfg.MODEL.DEVICE='cpu'
        self.clip_text_encoder = build_text_encoder(pretrain=True)
        self.clip_text_encoder.eval()

        self.predictor = DefaultPredictor(cfg)
        self.BUILDIN_CLASSIFIER = {
            'lvis': 'datasets/metadata/lvis_v1_clip_a+cname.npy',
            'objects365': 'datasets/metadata/o365_clip_a+cnamefix.npy',
            'openimages': 'datasets/metadata/oid_clip_a+cname.npy',
            'coco': 'datasets/metadata/coco_clip_a+cname.npy',
        }
        self.BUILDIN_METADATA_PATH = {
            'lvis': 'lvis_v1_val',
            'objects365': 'objects365_v2_val',
            'openimages': 'oid_val_expanded',
            'coco': 'coco_2017_val',
        }

    def predict(self,
          image: Path = Input(description="Grayscale input image"),
          predict_food_tray: bool = Input(description="Add Food Tray Prediction to output", default=False),
          vocabulary: str = Input(description="Vocabulary of choice", default='lvis', choices=['lvis', 'objects365', 'openimages', 'coco', 'custom']),
          custom_vocabulary: str = Input(description="Custom vocabulary, comma separated", default=None)
    ) -> Any:
        image = cv2.imread(str(image))
        if not vocabulary == 'custom':
            metadata = MetadataCatalog.get(self.BUILDIN_METADATA_PATH[vocabulary])
            classifier = self.BUILDIN_CLASSIFIER[vocabulary]
            num_classes = len(metadata.thing_classes)
            reset_cls_test(self.predictor.model, classifier, num_classes)

        else:
            assert custom_vocabulary is not None and len(custom_vocabulary.split(',')) > 0, \
                "Please provide your own vocabularies when vocabulary is set to 'custom'."
            metadata = MetadataCatalog.get(str(time.time()))
            metadata.thing_classes = custom_vocabulary.split(',')
            classifier = self.get_clip_embeddings(metadata.thing_classes)
            num_classes = len(metadata.thing_classes)
            reset_cls_test(self.predictor.model, classifier, num_classes)
            # Reset visualization threshold
            output_score_threshold = 0.3
            for cascade_stages in range(len(self.predictor.model.roi_heads.box_predictor)):
                self.predictor.model.roi_heads.box_predictor[cascade_stages].test_score_thresh = output_score_threshold

        outputs = self.predictor(image)
        v = Visualizer(image[:, :, ::-1], metadata)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out_path = Path(tempfile.mkdtemp()) / "out.png"
        cv2.imwrite(str(out_path), out.get_image()[:, :, ::-1])

        response = {}
        response['annotatedFile'] = out_path

        response['predictions'] = self.compute_full_prediction_output(metadata, outputs)

        if predict_food_tray:
            metadata = MetadataCatalog.get(str(time.time()))
            metadata.thing_classes = ["tray"]
            classifier = self.get_clip_embeddings(metadata.thing_classes)
            num_classes = len(metadata.thing_classes)
            reset_cls_test(self.predictor.model, classifier, num_classes)
            # Reset visualization threshold
            output_score_threshold = 0.3
            for cascade_stages in range(len(self.predictor.model.roi_heads.box_predictor)):
                self.predictor.model.roi_heads.box_predictor[cascade_stages].test_score_thresh = output_score_threshold
            outputs = self.predictor(image)

            response['tray_predictions'] = self.compute_full_prediction_output(metadata, outputs)

        return response


    def compute_full_prediction_output(self, metadata, outputs):
        predictions = []
        if "instances" in outputs:
            instances = outputs["instances"].to("cpu")
            class_names = metadata.thing_classes
            boxes = instances.pred_boxes if instances.has("pred_boxes") else None
            box_objs = []
            for b in  boxes.tensor.tolist():
                box_objs.append( { 'x1': b[0], 'y1': b[1], 'x2': b[2], 'y2': b[3] } )

            scores = instances.scores.tolist()
            predicted_class_indicies = instances.pred_classes.tolist() if instances.has("pred_classes") else None

            predicted_class_names = []
            for c_i in predicted_class_indicies:
                predicted_class_names.append(class_names[c_i])

            pred_masks = instances.pred_masks if instances.has("pred_masks") else None
            pixel_counts = torch.sum(pred_masks, dim=(1, 2)).tolist() if pred_masks != None else None

            for (c, score, box, pixel_count) in zip(predicted_class_names, scores, box_objs, pixel_counts):
                predictions.append({ 'class': c, 'score': score, 'box': box, 'pixelCount': pixel_count } )

        return predictions

    def get_clip_embeddings(self, vocabulary, prompt='a '):
        texts = [prompt + x for x in vocabulary]
        emb = self.clip_text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
        return emb
