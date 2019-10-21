from unittest import TestCase

from src.common.utils import *


class TestUtils(TestCase):
    def test_get_root_dir(self):
        self.assertTrue(get_root_dir().endswith("master-thesis"))

    def test_get_logging_dir(self):
        self.assertTrue(get_logs_dir(algorithm="policy").endswith("logs"))

    def test_get_tensorboard_dir(self):
        self.assertTrue(get_tensorboard_dir().endswith("tensorboard"))
