# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np

from grid2op.Reward.BaseReward import BaseReward
from grid2op.dtypes import dt_float

class NMinusOneReward(BaseReward):
    def __init__(self, rho_threshold=2):
        BaseReward.__init__(self)
        self.reward_min = None
        self.reward_max = None
        self.attackable_line_ids = None
        self.rho_threshold = dt_float(rho_threshold)

    def initialize(self, env):
        self.reward_min = dt_float(0)
        self.reward_max = dt_float(1)
        self.attackable_line_ids = list(range(env.n_line)) ##################################################

    def __call__(self,  action, env, has_error, is_done, is_illegal, is_ambiguous):
        if is_done:
            return self.reward_min

        obs = env.get_obs()
        rho_reward = np.maximum(self.rho_threshold - np.maximum(obs.rho, 1), 0)
        rho_reward /= self.rho_threshold - 1
        reward = dt_float(rho_reward.mean())

        return reward
