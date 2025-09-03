# Copyright (c) 2025, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings

import grid2op
from grid2op.Exceptions import EnvDependantAttributeCalledTooLate

# TODO detailed topo: test all the attributes: gen_uptime_lazy, gen_downtime_lazy, switches_state_lazy


class TestLazyObsAttributesLogicWorking(unittest.TestCase):
    def setUp(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox")
        return super().setUp()
    
    def tearDown(self):
        self.env.close()
        return super().tearDown()
    
    def test_ok(self):
        obs = self.env.reset(seed=0, options={"time serie id": 0})
        assert (obs.gen_uptime_lazy == [0, 0, 0, -1, -1, 0]).all()
        next_obs, *_ = self.env.step(self.env.action_space())
        assert (obs.gen_uptime_lazy == [0, 0, 0, -1, -1, 0]).all()
        assert (next_obs.gen_uptime_lazy == [1, 1, 1, -1, -1, 1]).all()
        next_obs2, *_ = self.env.step(self.env.action_space())
        assert (obs.gen_uptime_lazy == [0, 0, 0, -1, -1, 0]).all()
        assert (next_obs.gen_uptime_lazy == [1, 1, 1, -1, -1, 1]).all()
        assert (next_obs2.gen_uptime_lazy == [2, 2, 2, -1, -1, 2]).all()
        
    def test_ko_step(self):
        obs = self.env.reset(seed=0, options={"time serie id": 0})
        next_obs, *_ = self.env.step(self.env.action_space())
        assert (next_obs.gen_uptime_lazy == [1, 1, 1, -1, -1, 1]).all()
        with self.assertRaises(EnvDependantAttributeCalledTooLate):
            obs.gen_uptime_lazy
        next_obs2, *_ = self.env.step(self.env.action_space())
        with self.assertRaises(EnvDependantAttributeCalledTooLate):
            obs.gen_uptime_lazy
        assert (next_obs.gen_uptime_lazy == [1, 1, 1, -1, -1, 1]).all()
        assert (next_obs2.gen_uptime_lazy == [2, 2, 2, -1, -1, 2]).all()
        
    def test_ko_reset(self):
        obs = self.env.reset(seed=0, options={"time serie id": 0})
        next_obs, *_ = self.env.step(self.env.action_space())
        next_obs2, *_ = self.env.step(self.env.action_space())
        
        obs_next_ep = self.env.reset(seed=0, options={"time serie id": 1})
        with self.assertRaises(EnvDependantAttributeCalledTooLate):
            obs.gen_uptime_lazy
        with self.assertRaises(EnvDependantAttributeCalledTooLate):
            next_obs.gen_uptime_lazy
        with self.assertRaises(EnvDependantAttributeCalledTooLate):
            next_obs2.gen_uptime_lazy
        next_obs_next_ep, *_ = self.env.step(self.env.action_space())
        with self.assertRaises(EnvDependantAttributeCalledTooLate):
            obs.gen_uptime_lazy
        with self.assertRaises(EnvDependantAttributeCalledTooLate):
            next_obs.gen_uptime_lazy
        with self.assertRaises(EnvDependantAttributeCalledTooLate):
            next_obs2.gen_uptime_lazy
        next_obs2_next_ep, *_ = self.env.step(self.env.action_space())
        with self.assertRaises(EnvDependantAttributeCalledTooLate):
            obs.gen_uptime_lazy
        with self.assertRaises(EnvDependantAttributeCalledTooLate):
            next_obs.gen_uptime_lazy
        with self.assertRaises(EnvDependantAttributeCalledTooLate):
            next_obs2.gen_uptime_lazy
            
        obs_next_ep2 = self.env.reset(seed=0, options={"time serie id": 0})
        with self.assertRaises(EnvDependantAttributeCalledTooLate):
            obs.gen_uptime_lazy
        with self.assertRaises(EnvDependantAttributeCalledTooLate):
            next_obs.gen_uptime_lazy
        with self.assertRaises(EnvDependantAttributeCalledTooLate):
            next_obs2.gen_uptime_lazy
        next_obs_next_ep2, *_ = self.env.step(self.env.action_space())
        with self.assertRaises(EnvDependantAttributeCalledTooLate):
            obs.gen_uptime_lazy
        with self.assertRaises(EnvDependantAttributeCalledTooLate):
            next_obs.gen_uptime_lazy
        with self.assertRaises(EnvDependantAttributeCalledTooLate):
            next_obs2.gen_uptime_lazy
        next_obs2_next_ep2, *_ = self.env.step(self.env.action_space())
        with self.assertRaises(EnvDependantAttributeCalledTooLate):
            obs.gen_uptime_lazy
        with self.assertRaises(EnvDependantAttributeCalledTooLate):
            next_obs.gen_uptime_lazy
        with self.assertRaises(EnvDependantAttributeCalledTooLate):
            next_obs2.gen_uptime_lazy
            
    def test_ko_env_copy(self):
        obs = self.env.reset(seed=0, options={"time serie id": 0})
        next_obs, *_ = self.env.step(self.env.action_space())
        next_obs2, *_ = self.env.step(self.env.action_space())
        
        new_env = self.env.copy()
        with self.assertRaises(EnvDependantAttributeCalledTooLate):
            obs.gen_uptime_lazy
        with self.assertRaises(EnvDependantAttributeCalledTooLate):
            next_obs.gen_uptime_lazy
        # test I still can access attribute referenced by the
        # initial environment
        assert (next_obs2.gen_uptime_lazy == [2, 2, 2, -1, -1, 2]).all()
        
        obs_cpy = new_env.reset(seed=0, options={"time serie id": 0})
        next_obs_cpy, *_ = new_env.step(self.env.action_space())
        assert (next_obs_cpy.gen_uptime_lazy == [1, 1, 1, -1, -1, 1]).all()
        with self.assertRaises(EnvDependantAttributeCalledTooLate):
            obs_cpy.gen_uptime_lazy
        assert (next_obs2.gen_uptime_lazy == [2, 2, 2, -1, -1, 2]).all()
        next_obs2_cpy, *_ = new_env.step(self.env.action_space())
        with self.assertRaises(EnvDependantAttributeCalledTooLate):
            obs_cpy.gen_uptime_lazy
        assert (next_obs_cpy.gen_uptime_lazy == [1, 1, 1, -1, -1, 1]).all()
        assert (next_obs2_cpy.gen_uptime_lazy == [2, 2, 2, -1, -1, 2]).all()
        assert (next_obs2.gen_uptime_lazy == [2, 2, 2, -1, -1, 2]).all()
            
    def test_obs_copy(self):
        obs = self.env.reset(seed=0, options={"time serie id": 0})
        
        next_obs, *_ = self.env.step(self.env.action_space())
        assert (next_obs.gen_uptime_lazy == [1, 1, 1, -1, -1, 1]).all()
        with self.assertRaises(EnvDependantAttributeCalledTooLate):
            obs.gen_uptime_lazy
        obs_cpy = obs.copy()
        with self.assertRaises(EnvDependantAttributeCalledTooLate):
            obs_cpy.gen_uptime_lazy
        
        next_obs2, *_ = self.env.step(self.env.action_space())
        with self.assertRaises(EnvDependantAttributeCalledTooLate):
            obs.gen_uptime_lazy
        obs_cpy2 = obs.copy()
        with self.assertRaises(EnvDependantAttributeCalledTooLate):
            obs_cpy2.gen_uptime_lazy
            
        assert (next_obs.gen_uptime_lazy == [1, 1, 1, -1, -1, 1]).all()
        next_obs_cpy = next_obs.copy()
        assert (next_obs_cpy.gen_uptime_lazy == [1, 1, 1, -1, -1, 1]).all()
        
        assert (next_obs2.gen_uptime_lazy == [2, 2, 2, -1, -1, 2]).all()
        next_obs2_cpy = next_obs2.copy()
        assert (next_obs2_cpy.gen_uptime_lazy == [2, 2, 2, -1, -1, 2]).all()
        