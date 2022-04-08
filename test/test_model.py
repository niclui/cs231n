import unittest
import torch

from models import (CadeneModel,
                    TorchVisionModel,
                    EfficientNetModel,
                    Detectron2Model,
                    YOLOv3Model,
                    EfficientDetModel,
                    SMPModel)
from models import get_model
from util import Args

CLASSIFICATION_MODELS = [cls.__name__ for cls in
                         CadeneModel.__subclasses__() +
                         TorchVisionModel.__subclasses__() +
                         EfficientNetModel.__subclasses__()]

DETECTION_MODELS = [cls.__name__ for cls in
                    YOLOv3Model.__subclasses__() +
                    Detectron2Model.__subclasses__()]

SEGMENTATION_MODELS = [cls.__name__ for cls in
                       SMPModel.__subclasses__()]


MODELS = CLASSIFICATION_MODELS + DETECTION_MODELS + SEGMENTATION_MODELS


class TestModelMeta(type):
    def __new__(mcs, name, bases, dict):
        def gen_initialization_test(
                model_name,
                pretrained=False,
                num_classes=None):
            def test_model(self):
                args = Args({"model": model_name,
                             "pretrained": pretrained,
                             "gpus": None})
                if num_classes is not None:
                    args['num_classes'] = num_classes
                model = get_model(args)
                self.assertIsNotNone(model)

            return test_model

        def gen_segmentation_dimension_test(model_name,
                                            input_size,
                                            num_classes,
                                            pretrained="imagenet",
                                            encoder="resnet34", 
                                            ):
            def test_model(self):
                args = Args({"model": model_name,
                             "pretrained": pretrained,
                             "encoder": encoder, 
                             "gpus": None})
                if num_classes is not None:
                    args['num_classes'] = num_classes
                model = get_model(args)
                model.eval()
                output = model(torch.zeros(input_size))
                output_size = (
                    input_size[0],
                    num_classes,
                    input_size[2],
                    input_size[3])
                self.assertEqual(tuple(output.shape), output_size)

            return test_model

        def gen_classification_dimension_test(model_name,
                                              input_size,
                                              pretrained=False,
                                              num_classes=None,
                                              ):
            def test_model(self):
                args = Args({"model": model_name,
                             "pretrained": pretrained,
                             "gpus": None})
                if num_classes is not None:
                    args['num_classes'] = num_classes
                model = get_model(args)
                model.eval()
                output = model(torch.zeros(input_size))
                output_size = (input_size[0], num_classes)
                self.assertEqual(tuple(output.shape), output_size)

            return test_model

        for model in CLASSIFICATION_MODELS:
            dict[f"test_{model}_num_classes"] = gen_initialization_test(
                model, num_classes=1)
            dict[f"test_{model}_dimension"] = gen_classification_dimension_test(
                model, (2, 3, 256, 256), num_classes=5)
        
        for model in DETECTION_MODELS:
            dict[f"test_{model}_num_classes"] = gen_initialization_test(
                model, num_classes=1)

        for model in SEGMENTATION_MODELS:
            dict[f"test_{model}_dimension"] = gen_segmentation_dimension_test(
                model, (2, 3, 256, 256), num_classes=5)

        return type.__new__(mcs, name, bases, dict)


class TestModel(unittest.TestCase,
                metaclass=TestModelMeta):
    pass


if __name__ == "__main__":
    unittest.main(verbosity=0)
