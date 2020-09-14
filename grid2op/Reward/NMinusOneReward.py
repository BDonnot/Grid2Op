# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
import copy
import networkx as nx

from grid2op.dtypes import dt_float
from grid2op.Action._BackendAction import _BackendAction
from grid2op.Reward.BaseReward import BaseReward


class NMinusOneReward(BaseReward):
    def __init__(self):
        BaseReward.__init__(self)
        self.reward_min = None
        self.reward_max = None
        self.exp_top_x = 0.2
        self.exp_p = 0.95
        self.exp_lbda = None
        self.attackable_line_ids = None
        self.backend = None
        self.backend_action = None

    def initialize(self, env):
        self.reward_min = dt_float(0)
        self.reward_max = dt_float(1)

        if self.backend is None:
            self.backend = env.backend.copy()
            bk_act_cls = _BackendAction.init_grid(env.backend)
            self.backend_action = bk_act_cls()

        G = nx.Graph()
        nodes = np.arange(env.n_sub)
        edges = list(zip(env.line_or_to_subid, env.line_ex_to_subid))
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        bridges = list(nx.bridges(G))
        # consider all disconnections except the ones that would break the graph
        self.attackable_line_ids = [i for i, e in enumerate(edges) if e not in bridges]

        n = len(self.attackable_line_ids)
        self.exp_lbda = self._find_lambda(n, round(n * self.exp_top_x), self.exp_p)

    @staticmethod
    def _find_lambda(n, m, p, epsilon=1e-5):
        """ Dichotomy """
        lbda_min = 0
        lbda_max = 100
        while lbda_max - lbda_min > epsilon:
            lbda = (lbda_min + lbda_max) / 2
            weights = np.exp(np.linspace(0, 1, n) * -lbda)
            weights /= weights.sum()
            measure = weights[:m].sum()
            if measure > p:
                lbda_max = lbda
            else:
                lbda_min = lbda
    
        return lbda

    def __call__(self,  action, env, has_error, is_done, is_illegal, is_ambiguous):
        if is_done:
            return self.reward_min

        subrewards = [self.reward_max]
        self.backend.set_thermal_limit(env.get_thermal_limit())
        act = env.backend.get_action_to_set()
        for l_id in self.attackable_line_ids:
            this_n1 = copy.deepcopy(act)
            # this_n1.update({"set_line_status": [(l_id, -1)]})  # TODO fix that for the action
            # this_n1 += env.action_space({"set_line_status": [(l_id, -1)]})  # TODO fix that for the action
            self.backend_action += this_n1
            self.backend.apply_action(self.backend_action)
            self.backend._disconnect_line(l_id)
            try:
                # there is a bug in lightsimbackend that make it crash instead of diverging
                conv = self.backend.runpf()  # TODO add this test in the backend
            except:
                conv = False

            self.backend_action.reset()
            if conv:
                rho = self.backend.get_relative_flow()
                if np.any(rho > 1):
                    subrewards.append(self.reward_min)
                else:
                    subrewards.append(self.reward_max)
            else:
                # TODO study this, this makes it always go to O
                # probably: consider if there is an overflow on a n-1
                # then the score is O
                # if there is a divergence, score is -1
                # otherwise the score is 1
                # and at the end sum the scores
                subrewards.append(self.reward_min)

        subrewards = np.sort(subrewards)
        weights = np.exp(np.linspace(0, 1, len(subrewards)) * -self.exp_lbda)
        res = dt_float((subrewards * weights / weights.sum()).sum())

        return res
