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
    def __init__(self):
        BaseReward.__init__(self)
        self.reward_min = None
        self.reward_max = None
        self.attackable_line_ids = None

    def initialize(self, env):
        self.reward_min = dt_float(0)
        self.reward_max = dt_float(1)
        self.attackable_line_ids = np.arange(env.n_line)
        if env.name == 'l2rpn_neurips_2020_track1_warmup':
            self.attackable_line_ids = np.array([0, 9, 13, 14, 18, 23, 27, 39, 45, 56])

    def __call__(self,  action, env, has_error, is_done, is_illegal, is_ambiguous):
        if is_done:
            return self.reward_min

        subrewards = []
        for line_id in self.attackable_line_ids:
            this_backend = env.backend.copy()
            this_backend._disconnect_line(line_id)
            this_backend.next_grid_state(env)
            rho = this_backend.get_relative_flow()
            if np.isnan(rho).any():
                return self.reward_min

            sum_overflows = (rho > 1).sum()
            rho_reward = 1 - sum_overflows / env.n_line
            subrewards.append(rho_reward)

        return dt_float(np.min(subrewards))
