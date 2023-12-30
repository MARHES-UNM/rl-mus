import unittest
import numpy as np
from uav_sim.utils import utils

class TestUtils(unittest.TestCase):
    def setUp(self):
        pass

    def test_get_git_hash(self):
        print(f"git hash: {utils.get_git_hash()}")


if __name__ == "__main__":
    unittest.main()
