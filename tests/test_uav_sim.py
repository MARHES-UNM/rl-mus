import numpy as np
import unittest

# from tests import context
# import context
import unittest
from uav_sim.envs.uav_sim import UavSim


class TestUavSim(unittest.TestCase):
    def setUp(self):
        self.env = UavSim()

    # def test_lqr_controller(self):
    #     positions = np.array([[0.5, 0.5, 1], [0.5, 2, 2], [2, 0.5, 2], [2, 2, 1]])
    #     # positions = np.array([[4, 4, 1], [4, 2, 2], [4, 4, 2], [4, 4, 1]])
    #     des_pos = np.zeros((4, 12), dtype=np.float64)
    #     for idx in range(4):
    #         des_pos[idx, 0:3] = positions[idx, :]

    #         self.env.uavs[idx]._state[0:3] = positions[idx, :]
    #         # self.env.uavs[idx]._state[0] = positions[idx, 0]
    #         # self.env.uavs[idx]._state[1] = positions[idx, 1]
    #         # self.env.uavs[idx]._state[2] = positions[idx, 2]
    #     Ks = self.env.uavs[0].calc_k()

    #     actions = {}
    #     for i in range(100):
    #         for idx, pos in enumerate(des_pos):
    #             pos_er = pos - self.env.uavs[idx].state

    #             Ux = np.dot(Ks[0], pos_er[[0, 1, 8, 9]])[0]
    #             Uy = np.dot(Ks[0], pos_er[[2, 3, 6, 7]])[0]
    #             Uz = np.dot(Ks[2], pos_er[[4, 5]])[0]
    #             Uyaw = np.dot(Ks[3], pos_er[[10, 11]])[0]
    #             actions[idx] = np.array([Ux, Uy, Uz, Uyaw])
    #         self.env.step(actions)
    #         self.env.render()

        # K = self.env.uavs[0].calc_k()
        # actions = {}

        # for i in range(100):
        #     for idx, pos in enumerate(des_pos):
        #         cur_pos = self.env.uavs[idx].state
        #         actions[idx] = np.dot(-K, pos - cur_pos)
        #         # actions[i] = np.dot(K, )
        #     self.env.step(actions)
        #     self.env.render()

    # def test_setting_uav_pos(self):
    #     uav_pos = [[0.5, 0.5, 1], [0.5, 2, 2], [2, 0.5, 2], [2, 2, 1]]
    #     for idx, pos in enumerate(uav_pos):
    #         self.env.uavs[idx]._state[0] = pos[0]
    #         self.env.uavs[idx]._state[1] = pos[1]
    #         self.env.uavs[idx]._state[2] = pos[2]

    #     for i in range(20):
    #         actions = {_id: np.zeros(4) for _id in range(self.env.num_uavs)}
    #         self.env.step(actions)
    #         self.env.render()

    # def test_render(self):
    #     tf = 100
    #     t = 0
    #     actions = {}
    #     while t < tf:
    #         for i in range(self.env.num_uavs):
    #             actions[i] = np.ones(4) * self.env.uavs[i].m * self.env.uavs[i].g / 4
    #         self.env.step(actions)
    #         self.env.render()
    #         t += 1


if __name__ == "__main__":
    unittest.main()
