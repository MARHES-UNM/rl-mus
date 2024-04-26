import unittest
import numpy as np
from uav_sim.utils import utils


class TestUtils(unittest.TestCase):
    def setUp(self):
        pass

    def test_get_git_hash(self):
        print(f"git hash: {utils.get_git_hash()}")

    def test_max_diff(self):
        data = []
        output = utils.max_abs_diff(data)
        self.assertEqual(output, 0)

        data = [1]
        output = utils.max_abs_diff(data)
        self.assertEqual(output, 1)

        data = [0, 5]
        output = utils.max_abs_diff(data)
        self.assertEqual(output, 1)

        data = [1, 5, 8]
        output = utils.max_abs_diff(data)
        self.assertEqual(output, 7)


if __name__ == "__main__":
    unittest.main()
