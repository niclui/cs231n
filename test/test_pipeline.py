import os
import shutil
import unittest
from unittest import TestCase
from glob import glob
from os.path import join

from main import train as train_func
from main import test as test_func


class TestTrainFunc:
    PATH = os.path.abspath("./sandbox")
    
    def setUp(self):
        self.train = lambda x: train_func(exp_name=f"{self.TASK}_{x}",
                                save_dir=self.PATH,
                                gpus=None,
                                model=self.MODEL,
                                task=self.TASK,
                                weights_summary=None)


    def test_default(self):
        self.train("TEST_DEFAULT")

    def test_slurm(self):
        os.environ['SLURM_JOB_ID'] = "138"
        self.train("TEST_SLURM")
        self.assertTrue(os.path.exists(join(self.PATH,
                        f"{self.TASK}_TEST_SLURM/lightning_logs/version_0")))

    def test_override(self):
        self.train("TEST_OVERRIDE")
        with self.assertRaises(FileExistsError):
            self.train("TEST_OVERRIDE")


class TestClassificationTrainFunc(TestTrainFunc, TestCase):
    TASK = "classification" 
    MODEL = "ResNet18"
    

# Skip detection models due the limit of memory on CircleCI
# class TestDetectionTrainFunc(TestTrainFunc, TestCase):
#    TASK = "detection"
#    MODEL = "FasterRCNN"


class TestTestFunc:
    
    PATH = os.path.abspath("./sandbox")

    def setUp(self):
        self.exp_name = f"{self.TASK}_TEST_TESTING_DEFAULT"
        self.ckpt_path = join(self.PATH, f"{self.exp_name}/ckpts/epoch=0-step=0.ckpt") 
        
    def test_default(self):
        train_func(exp_name=self.exp_name,
                   save_dir=self.PATH,
                   gpus=None,
                   model=self.MODEL,
                   task=self.TASK,
                   weights_summary=None)

        test_func(ckpt_path=self.ckpt_path,
                  gpus=None)


class TestClassificationTestFunc(TestTestFunc, TestCase):
    TASK = "classification" 
    MODEL = "ResNet18"


class TestDetectionTestFunc(TestTestFunc, TestCase):
    TASK = "detection"
    MODEL = "FasterRCNN"
    
    
if __name__ == "__main__":
    unittest.main(verbosity=0)
    shutil.rmtree(os.path.abspath("./sandbox"), ignore_errors=True)
