# Copyright (c) 2023-2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings
import numpy as np
import hashlib

import grid2op
from grid2op.dtypes import dt_bool
from grid2op.Action import BaseAction, CompleteAction
from grid2op.Action._BackendAction import _BackendAction
from grid2op.Observation import BaseObservation
from grid2op.Runner import Runner
from grid2op.Backend import PandaPowerBackend
from grid2op.Space import AddDetailedTopoIEEE, DetailedTopoDescription
from grid2op.Agent import BaseAgent
from grid2op.Exceptions import AmbiguousAction
import pdb
REF_HASH = 'e5ccf7cbe54cb567eec33bfd738052f81dc5ac9a1ea2cd391d98f95f804a1273d0efac3d4e00aed9a43abf6ce8bf3fc3487a6c870bd6386dd7a84c3fa8344d99'


def _aux_test_correct(detailed_topo_desc : DetailedTopoDescription, gridobj, nb_bb_per_sub):
    if nb_bb_per_sub == 2:
        assert detailed_topo_desc is not None
        assert (detailed_topo_desc.load_to_conn_node_id == np.array([28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38], dtype=np.int32)).all()
        assert (detailed_topo_desc.gen_to_conn_node_id == np.array([39, 40, 41, 42, 43, 44], dtype=np.int32)).all()
        assert (detailed_topo_desc.line_or_to_conn_node_id == np.array([45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                                                                    56, 57, 58, 59, 60, 61, 62, 63, 64], dtype=np.int32)).all()
        
        # test the switches (but i don't want to copy this huge data here)
        assert (detailed_topo_desc.switches.sum(axis=0) == np.array([1159, 17732, 8730])).all()
        ref_1 = np.array([  1,   6,  11,  16,  21,  26,  31,  36,  41,  46,  51,  56,  61,
                           66, 117,  91,  92, 120,  95,  96, 123,  99, 100, 126, 103, 104,
                          129, 107, 108, 134, 117, 118, 137, 121, 122, 140, 125, 126, 143,
                          129, 130, 146, 133, 134, 149, 137, 138, 139, 102, 103, 142, 106,
                          107, 147, 116, 117, 149, 117, 118, 153, 124, 125, 148, 104, 105,
                          150, 105, 106, 152, 106, 107, 155, 110, 111, 157, 111, 112, 159,
                          112, 113, 162, 116, 117, 165, 120, 121, 169, 127, 128, 171, 128,
                          129, 173, 129, 130, 178, 139, 140, 180, 140, 141, 183, 144, 145,
                          187, 151, 152, 190, 155, 156, 183, 129, 130, 185, 130, 131, 188,
                          134, 135, 192, 141, 142, 196, 148, 149, 191, 128, 129, 196, 138,
                          139, 196, 133, 134, 199, 137, 138, 202, 141, 142, 203, 139, 140,
                          206, 143, 144, 214, 162, 163, 217, 166, 167, 220, 170, 171, 219,
                          162, 163, 225, 175, 176, 224, 167, 168, 228, 174, 175, 231, 178,
                          179, 226, 158, 159, 230, 165, 166, 229, 157, 158, 233, 164, 165,
                          234, 162, 163, 235, 160, 161, 239, 167, 168, 242, 171, 172])
        assert (detailed_topo_desc.switches.sum(axis=1) == ref_1).all()
        hash_ = hashlib.blake2b((detailed_topo_desc.switches.tobytes())).hexdigest()
        assert hash_ == REF_HASH, f"{hash_}"
    
    assert detailed_topo_desc.switches.shape[0] == (nb_bb_per_sub + 1) * (gridobj.dim_topo + gridobj.n_shunt) + gridobj.n_sub * (nb_bb_per_sub * (nb_bb_per_sub - 1) // 2)
    
    # test the names
    cls = type(detailed_topo_desc)
    dtd = detailed_topo_desc
    n_bb_per_sub = nb_bb_per_sub
    
    el_nm = "load"
    nb_el = gridobj.n_load
    prev_el = gridobj.n_sub * (nb_bb_per_sub * (nb_bb_per_sub - 1) // 2)
    for el_nm, nb_el in zip(["load", "gen", "line_or", "line_ex", "storage", "shunt"],
                            [gridobj.n_load, gridobj.n_gen, gridobj.n_line, gridobj.n_line, gridobj.n_storage, gridobj.n_shunt]):
        next_el = prev_el + nb_el * (1 + n_bb_per_sub) 
        for i, el in enumerate(dtd.conn_node_name[dtd.switches[prev_el : next_el : (1 + n_bb_per_sub), cls.CONN_NODE_1_ID_COL]]):
            assert f"conn_node_{el_nm}_{i}" in el, f"error for what should be the switch connecting conn node to {el_nm} {i} to its conn node breaker"
        for i, el in enumerate(dtd.conn_node_name[dtd.switches[prev_el : next_el : (1 + n_bb_per_sub), cls.CONN_NODE_2_ID_COL]]):
            assert f"conn_node_breaker_{el_nm}_{i}" in el, f"error for what should be the switch connecting conn node to {el_nm} {i} to its conn node breaker"
        
        for bb_i in range(1, n_bb_per_sub + 1):
            assert (dtd.conn_node_name[dtd.switches[prev_el : next_el : (1 + n_bb_per_sub), cls.CONN_NODE_2_ID_COL]] == 
                    dtd.conn_node_name[dtd.switches[(prev_el + bb_i) : next_el : (1 + n_bb_per_sub), cls.CONN_NODE_1_ID_COL]]).all(), (
                        f"Error for what should connect a {el_nm} breaker connection node to busbar {bb_i}")
            
            for i, el in enumerate(dtd.conn_node_name[dtd.switches[(prev_el + bb_i) : next_el : (1 + n_bb_per_sub), cls.CONN_NODE_2_ID_COL]]):
                assert f"busbar_{bb_i-1}" in el, f"error for what should be the switch connecting conn node {el_nm} {i} (its breaker) to busbar {bb_i}"
        prev_el = next_el      
         
    # siwtches to pos topo vect
    # TODO detailed topo
    # ref_switches_pos_topo_vect = np.array([ 2,  2,  0,  0,  1,  1,  8,  8,  7,  7,  4,  4,  5,  5,  6,  6,  3,
    #                                         3, 12, 12, 11, 11, 10, 10,  9,  9, 18, 18, 15, 15, 16, 16, 17, 17,
    #                                        13, 13, 14, 14, 23, 23, 22, 22, 19, 19, 20, 20, 21, 21, 30, 30, 28,
    #                                        28, 29, 29, 24, 24, 25, 25, 26, 26, 27, 27, 31, 31, 33, 33, 32, 32,
    #                                        34, 34, 36, 36, 35, 35, 37, 37, 42, 42, 38, 38, 39, 39, 41, 41, 40,
    #                                        40, -1, -1, 45, 45, 44, 44, 43, 43, 48, 48, 46, 46, 47, 47, 51, 51,
    #                                        50, 50, 49, 49, 55, 55, 54, 54, 52, 52, 53, 53, 58, 58, 56, 56, 57,
    #                                        57], dtype=np.int32)
    # for i in range(-1, dim_topo):
    #     assert np.sum(ref_switches_pos_topo_vect == i).sum() == 2, f"error for topo_vect_id = {i}"
    # assert np.all(detailed_topo_desc.switches_to_topovect_id == ref_switches_pos_topo_vect)
    

class _PPBkForTestDetTopo(AddDetailedTopoIEEE, PandaPowerBackend):
    pass


class TestDTDAgent(BaseAgent):
    def act(self, observation: BaseObservation, reward: float, done: bool = False) -> BaseAction:
        _aux_test_correct(type(observation).detailed_topo_desc, type(observation),  type(observation).n_busbar_per_sub)
        return super().act(observation, reward, done)

        
class DetailedTopoTester(unittest.TestCase):
    def _aux_n_bb_per_sub(self):
        return 2
    
    def setUp(self) -> None:
        n_bb_per_sub = self._aux_n_bb_per_sub()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
                        "educ_case14_storage",
                        n_busbar=n_bb_per_sub,
                        test=True,
                        backend=_PPBkForTestDetTopo(),
                        action_class=CompleteAction,
                        _add_to_name=f"{type(self).__name__}_{n_bb_per_sub}",
                    )
        if isinstance(self, DetailedTopoTester):
            # weird hack I am doing: I reuse the same method
            # from another class
            # to initialize the same env, but by doing so, I cannot
            # "super"
            return super().setUp()
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_init_ok(self):
        obs = self.env.reset()
        _aux_test_correct(type(obs).detailed_topo_desc, type(obs), self._aux_n_bb_per_sub())

    def test_work_simulate(self):
        obs = self.env.reset()
        _aux_test_correct(type(obs).detailed_topo_desc, type(obs), self._aux_n_bb_per_sub())
        sim_o, *_ = obs.simulate(self.env.action_space())
        _aux_test_correct(type(sim_o).detailed_topo_desc, type(sim_o), self._aux_n_bb_per_sub())
    
    def test_runner_seq(self):
        obs = self.env.reset()
        runner = Runner(**self.env.get_params_for_runner(), agentClass=TestDTDAgent)
        runner.run(nb_episode=1, max_iter=10)
        runner.run(nb_episode=1, max_iter=10, add_detailed_output=True)
    
    def test_runner_par(self):
        obs = self.env.reset()
        runner = Runner(**self.env.get_params_for_runner(), agentClass=TestDTDAgent)
        runner.run(nb_episode=2, max_iter=10, nb_process=2)
        runner.run(nb_episode=2, max_iter=10, add_detailed_output=True, nb_process=2)
    
    def test_env_cpy(self):
        obs = self.env.reset()
        env_cpy = self.env.copy()
        obs_cpy = env_cpy.reset()
        _aux_test_correct(type(obs_cpy).detailed_topo_desc, type(obs_cpy), self._aux_n_bb_per_sub())
    
    def test_compute_switches_position(self):
        nb_busbar = self._aux_n_bb_per_sub()
        start_id = (nb_busbar * (nb_busbar - 1) // 2) * type(self.env).n_sub
        
        obs = self.env.reset()
        dtd = type(obs).detailed_topo_desc
        switches_state = dtd.compute_switches_position(obs.topo_vect, obs._shunt_bus)
        assert switches_state.sum() == 120
        assert switches_state[start_id::(nb_busbar + 1)].all()  # all connected
        assert switches_state[(start_id + 1)::(nb_busbar + 1)].all()  # all on bus 1
        assert (~switches_state[(start_id + 2)::(nb_busbar + 1)]).all()  # nothing on busbar 2
        
        # move everything to bus 2
        switches_state = dtd.compute_switches_position(np.full(obs.topo_vect.shape, fill_value=2),
                                                                                np.full(obs._shunt_bus.shape, fill_value=2))
        assert switches_state.sum() == 120
        assert switches_state[start_id::(nb_busbar + 1)].all()  # all connected
        assert switches_state[(start_id + 2)::(nb_busbar + 1)].all()  # all on busbar 2
        assert (~switches_state[(start_id + 1)::(nb_busbar + 1)]).all()  # nothing on busbar 1
        
        # now check some disconnected elements (*eg* line id 0)
        topo_vect = 1 * obs.topo_vect
        topo_vect[type(obs).line_or_pos_topo_vect[0]] = -1
        topo_vect[type(obs).line_ex_pos_topo_vect[0]] = -1
        switches_state = dtd.compute_switches_position(topo_vect, obs._shunt_bus)
        # quickly check other elements
        assert switches_state.sum() == 116
        assert switches_state[start_id::(nb_busbar + 1)].sum() == 58
        assert switches_state[(start_id + 1)::(nb_busbar + 1)].sum() == 58  # busbar 1
        assert switches_state[(start_id + 2)::(nb_busbar + 1)].sum() == 0  # busbar 2
        id_switch_or = dtd.get_switch_id_ieee(dtd.line_or_to_conn_node_id[0])
        id_switch_ex = dtd.get_switch_id_ieee(dtd.line_ex_to_conn_node_id[0])
        assert (~switches_state[id_switch_or:(id_switch_or + nb_busbar + 1)]).all()
        assert (~switches_state[id_switch_ex:(id_switch_ex + nb_busbar + 1)]).all()
        
        # and now elements per elements
        # load 3 to bus 2
        topo_vect = 1 * obs.topo_vect
        topo_vect[type(obs).load_pos_topo_vect[3]] = 2
        switches_state = dtd.compute_switches_position(topo_vect, obs._shunt_bus)
        assert switches_state.sum() == 120
        assert switches_state[start_id::(nb_busbar + 1)].sum() == 60
        assert switches_state[(start_id + 1)::(nb_busbar + 1)].sum() == 59  # busbar 1
        assert switches_state[(start_id + 2)::(nb_busbar + 1)].sum() == 1  # busbar 2
        id_switch = dtd.get_switch_id_ieee(dtd.load_to_conn_node_id[3])
        assert switches_state[id_switch:(id_switch + nb_busbar + 1)].sum() == 2  # only 2 switches closed
        assert switches_state[id_switch + 2]  # busbar 2
        
        # gen 1 to bus 2
        topo_vect = 1 * obs.topo_vect
        topo_vect[type(obs).gen_pos_topo_vect[1]] = 2
        switches_state = dtd.compute_switches_position(topo_vect, obs._shunt_bus)
        assert switches_state.sum() == 120
        assert switches_state[start_id::(nb_busbar + 1)].sum() == 60
        assert switches_state[(start_id + 1)::(nb_busbar + 1)].sum() == 59  # busbar 1
        assert switches_state[(start_id + 2)::(nb_busbar + 1)].sum() == 1  # busbar 2
        id_switch = dtd.get_switch_id_ieee(dtd.gen_to_conn_node_id[1])
        assert switches_state[id_switch:(id_switch + nb_busbar + 1)].sum() == 2  # only 2 switches closed
        assert switches_state[id_switch + 2]  # busbar 2
        
        # line or 6 to bus 2
        topo_vect = 1 * obs.topo_vect
        el_id = 6
        topo_vect[type(obs).line_or_pos_topo_vect[el_id]] = 2
        switches_state = dtd.compute_switches_position(topo_vect, obs._shunt_bus)
        assert switches_state.sum() == 120
        assert switches_state[start_id::(nb_busbar + 1)].sum() == 60
        assert switches_state[(start_id + 1)::(nb_busbar + 1)].sum() == 59  # busbar 1
        assert switches_state[(start_id + 2)::(nb_busbar + 1)].sum() == 1  # busbar 2
        id_switch = dtd.get_switch_id_ieee(dtd.line_or_to_conn_node_id[el_id])
        assert switches_state[id_switch:(id_switch + nb_busbar + 1)].sum() == 2  # only 2 switches closed
        assert switches_state[id_switch + 2]  # busbar 2
        
        # line ex 9 to bus 2
        topo_vect = 1 * obs.topo_vect
        el_id = 9
        topo_vect[type(obs).line_ex_pos_topo_vect[el_id]] = 2
        switches_state = dtd.compute_switches_position(topo_vect, obs._shunt_bus)
        assert switches_state.sum() == 120
        assert switches_state[start_id::(nb_busbar + 1)].sum() == 60
        assert switches_state[(start_id + 1)::(nb_busbar + 1)].sum() == 59  # busbar 1
        assert switches_state[(start_id + 2)::(nb_busbar + 1)].sum() == 1  # busbar 2
        id_switch = dtd.get_switch_id_ieee(dtd.line_ex_to_conn_node_id[el_id])
        assert switches_state[id_switch:(id_switch + nb_busbar + 1)].sum() == 2  # only 2 switches closed
        assert switches_state[id_switch + 2]  # busbar 2
        
        # storage 0 to bus 2
        topo_vect = 1 * obs.topo_vect
        el_id = 0
        topo_vect[type(obs).storage_pos_topo_vect[el_id]] = 2
        switches_state = dtd.compute_switches_position(topo_vect, obs._shunt_bus)
        assert switches_state.sum() == 120
        assert switches_state[start_id::(nb_busbar + 1)].sum() == 60
        assert switches_state[(start_id + 1)::(nb_busbar + 1)].sum() == 59  # busbar 1
        assert switches_state[(start_id + 2)::(nb_busbar + 1)].sum() == 1  # busbar 2
        id_switch = dtd.get_switch_id_ieee(dtd.storage_to_conn_node_id[el_id])
        assert switches_state[id_switch:(id_switch + nb_busbar + 1)].sum() == 2  # only 2 switches closed
        assert switches_state[id_switch + 2]  # busbar 2
        
        # shunt 0 to bus 2
        shunt_bus = 1 * obs._shunt_bus
        el_id = 0
        shunt_bus[el_id] = 2
        switches_state = dtd.compute_switches_position(obs.topo_vect, shunt_bus)
        assert switches_state.sum() == 120
        assert switches_state[start_id::(nb_busbar + 1)].sum() == 60
        assert switches_state[(start_id + 1)::(nb_busbar + 1)].sum() == 59  # busbar 1
        assert switches_state[(start_id + 2)::(nb_busbar + 1)].sum() == 1  # busbar 2
        id_switch = dtd.get_switch_id_ieee(dtd.shunt_to_conn_node_id[el_id])
        assert switches_state[id_switch:(id_switch + nb_busbar + 1)].sum() == 2  # only 2 switches closed
        assert switches_state[id_switch + 2]  # busbar 2
    
    def test_get_all_switches(self):
        """test I can use bkact.get_all_switches"""
        nb_busbar = self._aux_n_bb_per_sub()
        start_id = (nb_busbar * (nb_busbar - 1) // 2) * type(self.env).n_sub
        
        obs = self.env.reset()
        bk_act = self.env._backend_action
        dtd = type(obs).detailed_topo_desc
        
        # nothing modified
        switches_state = bk_act.get_all_switches()
        assert switches_state.sum() == 120
        assert switches_state[start_id::(nb_busbar + 1)].all()  # all connected
        assert switches_state[(start_id + 1)::(nb_busbar + 1)].all()  # all on bus 1
        assert (~switches_state[(start_id + 2)::(nb_busbar + 1)]).all()  # nothing on busbar 2
        
        # I modified the position of a "regular" element load 1 for the sake of the example
        bk_act += self.env.action_space({"set_bus": {"loads_id": [(1, 2)]}})
        switches_state = bk_act.get_all_switches()
        assert switches_state.sum() == 120
        assert switches_state[start_id::(nb_busbar + 1)].sum() == 60
        assert switches_state[(start_id + 1)::(nb_busbar + 1)].sum() == 59  # busbar 1
        assert switches_state[(start_id + 2)::(nb_busbar + 1)].sum() == 1  # busbar 2
        id_switch = dtd.get_switch_id_ieee(dtd.load_to_conn_node_id[1])
        assert switches_state[id_switch:(id_switch + nb_busbar + 1)].sum() == 2  # only 2 switches closed
        assert switches_state[id_switch + 2]  # busbar 2
                
        # I disconnect it
        bk_act += self.env.action_space({"set_bus": {"loads_id": [(1, -1)]}})
        switches_state = bk_act.get_all_switches()
        assert switches_state.sum() == 118
        assert switches_state[start_id::(nb_busbar + 1)].sum() == 59
        assert switches_state[(start_id + 1)::(nb_busbar + 1)].sum() == 59  # busbar 1
        assert switches_state[(start_id + 2)::(nb_busbar + 1)].sum() == 0  # busbar 2
        id_switch = dtd.get_switch_id_ieee(dtd.load_to_conn_node_id[1])
        assert switches_state[id_switch:(id_switch + nb_busbar + 1)].sum() == 0  # only 2 switches closed
        bk_act += self.env.action_space({"set_bus": {"loads_id": [(1, 1)]}})
        
        # I modify the position of a shunt (a bit special)
        bk_act += self.env.action_space({"shunt": {"set_bus": [(0, 2)]}})
        switches_state = bk_act.get_all_switches()
        assert switches_state.sum() == 120
        assert switches_state[start_id::(nb_busbar + 1)].sum() == 60
        assert switches_state[(start_id + 1)::(nb_busbar + 1)].sum() == 59  # busbar 1
        assert switches_state[(start_id + 2)::(nb_busbar + 1)].sum() == 1  # busbar 2
        id_switch = dtd.get_switch_id_ieee(dtd.shunt_to_conn_node_id[0])
        assert switches_state[id_switch:(id_switch + nb_busbar + 1)].sum() == 2  # only 2 switches closed
        assert switches_state[id_switch + 2]  # busbar 2
        
        # I disconnect it
        bk_act += self.env.action_space({"shunt": {"set_bus": [(0, -1)]}})
        switches_state = bk_act.get_all_switches()
        assert switches_state.sum() == 118
        assert switches_state[start_id::(nb_busbar + 1)].sum() == 59
        assert switches_state[(start_id + 1)::(nb_busbar + 1)].sum() == 59  # busbar 1
        assert switches_state[(start_id + 2)::(nb_busbar + 1)].sum() == 0  # busbar 2
        id_switch = dtd.get_switch_id_ieee(dtd.shunt_to_conn_node_id[0])
        assert switches_state[id_switch:(id_switch + nb_busbar + 1)].sum() == 0  # only 2 switches closed
        
        # set back it back to its original position
        bk_act += self.env.action_space({"shunt": {"set_bus": [(0, 1)]}})
        switches_state = bk_act.get_all_switches()
        assert switches_state.sum() == 120
        assert switches_state[start_id::(nb_busbar + 1)].sum() == 60
        assert switches_state[(start_id + 1)::(nb_busbar + 1)].sum() == 60  # busbar 1
        assert switches_state[(start_id + 2)::(nb_busbar + 1)].sum() == 0   # busbar 2
        id_switch = dtd.get_switch_id_ieee(dtd.shunt_to_conn_node_id[0])
        assert switches_state[id_switch:(id_switch + nb_busbar + 1)].sum() == 2  # only 2 switches closed
        assert switches_state[id_switch + 1]  # busbar 1
        
        # then I disconnect a powerline (check that both ends are disconnected)
        bk_act += self.env.action_space({"set_bus": {"lines_or_id": [(3, -1)]}})
        switches_state = bk_act.get_all_switches()
        assert switches_state.sum() == 116
        assert switches_state[start_id::(nb_busbar + 1)].sum() == 58
        assert switches_state[(start_id + 1)::(nb_busbar + 1)].sum() == 58  # busbar 1
        assert switches_state[(start_id + 2)::(nb_busbar + 1)].sum() == 0  # busbar 2
        id_switch_or = dtd.get_switch_id_ieee(dtd.line_or_to_conn_node_id[3])
        id_switch_ex = dtd.get_switch_id_ieee(dtd.line_ex_to_conn_node_id[3])
        assert (~switches_state[id_switch_or:(id_switch_or + nb_busbar + 1)]).all()
        assert (~switches_state[id_switch_ex:(id_switch_ex + nb_busbar + 1)]).all()
        
    def test_from_switches_position_basic(self):
        nb_busbar = self._aux_n_bb_per_sub()
        start_id = (nb_busbar * (nb_busbar - 1) // 2) * type(self.env).n_sub
        dtd = type(self.env).detailed_topo_desc
        
        # all connected
        switches_state = np.ones(dtd.switches.shape[0], dtype=dt_bool)
        topo_vect, shunt_bus = dtd.from_switches_position(switches_state)
        assert (topo_vect == 1).all()
        assert (shunt_bus == 1).all()
        
        # connect all to bus 1
        switches_state = np.ones(dtd.switches.shape[0], dtype=dt_bool)
        switches_state[:start_id] = False
        for bb_id in range(1, nb_busbar + 1):
            if bb_id == 1:
                # busbar 1
                switches_state[(start_id + bb_id)::(nb_busbar + 1)] = True  
            else:
                # busbar 2 or more 
                switches_state[(start_id + bb_id)::(nb_busbar + 1)] = False
        topo_vect, shunt_bus = dtd.from_switches_position(switches_state)
        assert (topo_vect == 1).all()
        assert (shunt_bus == 1).all()
        
        # connect all to bus 2
        switches_state = np.ones(dtd.switches.shape[0], dtype=dt_bool)
        switches_state[:start_id] = False
        for bb_id in range(1, nb_busbar + 1):
            if bb_id == 2:
                # busbar 2
                switches_state[(start_id + bb_id)::(nb_busbar + 1)] = True  
            else:
                # other busbars
                switches_state[(start_id + bb_id)::(nb_busbar + 1)] = False
        topo_vect, shunt_bus = dtd.from_switches_position(switches_state)
        assert (topo_vect == 2).all()
        assert (shunt_bus == 2).all()
    
        # connect all el to busbar 2, but connect all busbar together
        switches_state = np.ones(dtd.switches.shape[0], dtype=dt_bool)
        switches_state[:start_id] = True  # connect all busbars together
        for bb_id in range(1, nb_busbar + 1):
            if bb_id == 2:
                # busbar 2
                switches_state[(start_id + bb_id)::(nb_busbar + 1)] = True  
            else:
                # other busbars
                switches_state[(start_id + bb_id)::(nb_busbar + 1)] = False
        topo_vect, shunt_bus = dtd.from_switches_position(switches_state)
        assert (topo_vect == 1).all()
        assert (shunt_bus == 1).all()
        
        # connect all el to busbar 1, but disconnect the element with their breaker
        switches_state = np.ones(dtd.switches.shape[0], dtype=dt_bool)
        switches_state[:start_id] = True  # connect all busbars together
        switches_state[(start_id)::(nb_busbar + 1)] = False  # breaker
        for bb_id in range(1, nb_busbar + 1):
            if bb_id == 2:
                # busbar 2
                switches_state[(start_id + bb_id)::(nb_busbar + 1)] = True  
            else:
                # other busbars
                switches_state[(start_id + bb_id)::(nb_busbar + 1)] = False
        topo_vect, shunt_bus = dtd.from_switches_position(switches_state)
        assert (topo_vect == -1).all()
        assert (shunt_bus == -1).all()
        
    def test_from_switches_position_more_advanced(self):
        nb_busbar = self._aux_n_bb_per_sub()
        start_id = (nb_busbar * (nb_busbar - 1) // 2) * type(self.env).n_sub
        dtd = type(self.env).detailed_topo_desc
        
        # if you change the env it will change...
        sub_id = 1
        mask_el_this = type(self.env).grid_objects_types[:,0] == sub_id
        load_this = [0]
        gen_this = [0]
        line_or_this = [2]  # , 3, 4]
        line_ex_this = [0]
        
        bbs_switch_bb1_bb2 = sub_id * (nb_busbar * (nb_busbar - 1) // 2)  # switch between busbar 1 and busbar 2 at this substation
        load_id_switch = dtd.get_switch_id_ieee(dtd.load_to_conn_node_id[load_this[0]])
        gen_id_switch = dtd.get_switch_id_ieee(dtd.gen_to_conn_node_id[gen_this[0]])
        lor_id_switch = dtd.get_switch_id_ieee(dtd.line_or_to_conn_node_id[line_or_this[0]])
        lex_id_switch = dtd.get_switch_id_ieee(dtd.line_ex_to_conn_node_id[line_ex_this[0]])
        
        el_id_switch = load_id_switch
        el_this = load_this
        vect_topo_vect = type(self.env).load_pos_topo_vect
        for el_id_switch, el_this, vect_topo_vect, tag in zip([load_id_switch, gen_id_switch, lor_id_switch, lex_id_switch],
                                                              [load_this, gen_this, line_or_this, line_ex_this],
                                                              [type(self.env).load_pos_topo_vect, 
                                                               type(self.env).gen_pos_topo_vect,
                                                               type(self.env).line_or_pos_topo_vect,
                                                               type(self.env).line_ex_pos_topo_vect],
                                                              ["load", "gen", "lor", "lex"]):
            # all connected
            switches_state = np.ones(dtd.switches.shape[0], dtype=dt_bool)
            switches_state[:start_id] = False # deactivate all busbar coupler
            # assign all element to busbar 1
            for bb_id in range(1, nb_busbar + 1):
                if bb_id == 1:
                    # busbar 1
                    switches_state[(start_id + bb_id)::(nb_busbar + 1)] = True  
                else:
                    # other busbars
                    switches_state[(start_id + bb_id)::(nb_busbar + 1)] = False
                    
            # disconnect the load with the breaker
            switches_state[el_id_switch] = False
            topo_vect, shunt_bus = dtd.from_switches_position(switches_state)
            assert topo_vect[vect_topo_vect[el_this]] == -1, f"error for {tag}"
            assert (topo_vect == 1).sum() == 58, f"error for {tag}"
            switches_state[el_id_switch] = True
            
            # disconnect the load by disconnecting it of all the busbars
            switches_state[(el_id_switch + 1):(el_id_switch + nb_busbar +1)] = False
            topo_vect, shunt_bus = dtd.from_switches_position(switches_state)
            assert topo_vect[vect_topo_vect[el_this]] == -1, f"error for {tag}"
            assert (topo_vect == 1).sum() == 58, f"error for {tag}"
            switches_state[(el_id_switch + 1)] = True  # busbar 1
            
            # now connect the load to busbar 2
            switches_state[(el_id_switch + 1)] = False  # busbar 1
            switches_state[(el_id_switch + 2)] = True  # busbar 2
            topo_vect, shunt_bus = dtd.from_switches_position(switches_state)
            assert topo_vect[vect_topo_vect[el_this]] == 2, f"error for {tag}"
            assert (topo_vect == 1).sum() == 58, f"error for {tag}"
            
            # load still on busbar 2, but disconnected
            switches_state[(el_id_switch)] = False  # busbar 1
            topo_vect, shunt_bus = dtd.from_switches_position(switches_state)
            assert topo_vect[vect_topo_vect[el_this]] == -1, f"error for {tag}"
            assert (topo_vect == 1).sum() == 58, f"error for {tag}"
            # reset to orig state
            switches_state[(el_id_switch)] = True  # busbar 1
            switches_state[(el_id_switch + 1)] = True  # busbar 1
            switches_state[(el_id_switch + 2)] = False  # busbar 2
            
            # load on busbar 2, but busbars connected
            switches_state[(el_id_switch + 1)] = False  # busbar 1
            switches_state[(el_id_switch + 2)] = True   # busbar 2
            switches_state[bbs_switch_bb1_bb2] = True     # switch between busbars
            topo_vect, shunt_bus = dtd.from_switches_position(switches_state)
            assert topo_vect[vect_topo_vect[el_this]] == 1, f"error for {tag}"
            assert (topo_vect == 1).sum() == 59, f"error for {tag}"
    
    # TODO detailed topo : test_from_switches_position_more_advanced_shunt (shunts not tested above)
    # TODO detailed topo add more tests
    
    
class DetailedTopoTester_3bb(DetailedTopoTester):
    def _aux_n_bb_per_sub(self):
        return 3
    

class DetailedTopoTester_Action(unittest.TestCase):
    def _aux_n_bb_per_sub(self):
        return 2
    
    def setUp(self) -> None:
        DetailedTopoTester.setUp(self)
        self.li_flag_nm = [
            "_modif_inj",
            "_modif_set_bus",
            "_modif_change_bus",
            "_modif_set_status",
            "_modif_change_status",
            "_modif_redispatch",
            "_modif_storage",
            "_modif_curtailment",
            "_modif_alarm",
            "_modif_alert",
            "_modif_set_switch",
            "_modif_change_switch",
        ]
        type(self.env.action_space._template_obj).ISSUE_WARNING_SWITCH_SET_CHANGE = "never"
        return super().setUp()    
    
    def test_can_do_set(self):
        act = self.env.action_space({"set_switch": [(0, 1)]})
        assert act._modif_set_switch
        for flag_nm in self.li_flag_nm:
            if flag_nm == "_modif_set_switch":
                continue
            assert not getattr(act, flag_nm)
        assert act._set_switch_status[0] == 1
        assert (act._set_switch_status[1:] == 0).all()
        
        act = self.env.action_space({"set_switch": [(1, -1)]})
        assert act._modif_set_switch
        assert act._set_switch_status[1] == -1
        assert (act._set_switch_status[0] == 0).all()
        assert (act._set_switch_status[2:] == 0).all()
        
        # with the property
        act = self.env.action_space()
        act.set_switch = [(0, 1)]
        assert act._modif_set_switch
        for flag_nm in self.li_flag_nm:
            if flag_nm == "_modif_set_switch":
                continue
            assert not getattr(act, flag_nm)
        assert act._set_switch_status[0] == 1
        assert (act._set_switch_status[1:] == 0).all()
        
        act = self.env.action_space()
        act.set_switch = [(1, -1)]
        assert act._modif_set_switch
        assert act._set_switch_status[1] == -1
        assert (act._set_switch_status[0] == 0).all()
        assert (act._set_switch_status[2:] == 0).all()
        
    def test_can_do_change(self):
        act = self.env.action_space({"change_switch": [0]})
        assert act._modif_change_switch
        for flag_nm in self.li_flag_nm:
            if flag_nm == "_modif_change_switch":
                continue
            assert not getattr(act, flag_nm)
        assert act._change_switch_status[0]
        assert (~act._change_switch_status[1:]).all()
        # with the property
        act = self.env.action_space()
        act.change_switch = [0]
        assert act._modif_change_switch
        for flag_nm in self.li_flag_nm:
            if flag_nm == "_modif_change_switch":
                continue
            assert not getattr(act, flag_nm)
        assert act._change_switch_status[0]
        assert (~act._change_switch_status[1:]).all()
        
    def test_ambiguous_set_switch(self):
        with self.assertRaises(AmbiguousAction):
            act = self.env.action_space({"set_switch": [(-1, 1)]})
        with self.assertRaises(AmbiguousAction):
            act = self.env.action_space({"set_switch": [(type(self.env).detailed_topo_desc.switches.shape[0], 1)]})
        with self.assertRaises(AmbiguousAction):
            act = self.env.action_space({"set_switch": [(0, -2)]})
        with self.assertRaises(AmbiguousAction):
            act = self.env.action_space({"set_switch": [(0, 2)]})
            
        # same sub with set_bus and set switch
        act = self.env.action_space()
        nb_bb = self._aux_n_bb_per_sub()
        act.set_switch = [ (nb_bb * (nb_bb - 1) // 2, + 1)]
        act.load_set_bus = [(0, 1)]
        with self.assertRaises(AmbiguousAction):
            act._check_for_ambiguity()
            
        # same sub with change_bus and set switch
        act = self.env.action_space()
        nb_bb = self._aux_n_bb_per_sub()
        act.set_switch = [ (nb_bb * (nb_bb - 1) // 2, + 1)]
        act.load_change_bus = [0]
        with self.assertRaises(AmbiguousAction):
            act._check_for_ambiguity()
        
        # set switch and change switch
        act = self.env.action_space()
        act.set_switch = [(0, 1)]
        act.change_switch = [0]
        with self.assertRaises(AmbiguousAction):
            act._check_for_ambiguity()
            
    def test_ambiguous_change_switch(self):
        with self.assertRaises(AmbiguousAction):
            act = self.env.action_space({"change_switch": [-1]})
        with self.assertRaises(AmbiguousAction):
            act = self.env.action_space({"change_switch": [type(self.env).detailed_topo_desc.switches.shape[0]]})
            
        # same sub with set_bus and set switch
        act = self.env.action_space()
        nb_bb = self._aux_n_bb_per_sub()
        act.change_switch = [nb_bb * (nb_bb - 1) // 2]
        act.load_set_bus = [(0, 1)]
        with self.assertRaises(AmbiguousAction):
            act._check_for_ambiguity()
            
        # same sub with change_bus and set switch
        act = self.env.action_space()
        nb_bb = self._aux_n_bb_per_sub()
        act.change_switch = [nb_bb * (nb_bb - 1) // 2]
        act.load_change_bus = [0]
        with self.assertRaises(AmbiguousAction):
            act._check_for_ambiguity()
        
        # set switch and change switch
        act = self.env.action_space()
        act.set_switch = [(0, 1)]
        act.change_switch = [0]
        with self.assertRaises(AmbiguousAction):
            act._check_for_ambiguity()
            
    def test_backend_action_set_switch(self):
        nb_busbar = self._aux_n_bb_per_sub()
        dtd = type(self.env).detailed_topo_desc
        
        # if you change the env it will change...
        sub_id = 1
        load_this = [0]
        gen_this = [0]
        line_or_this = [2]
        line_ex_this = [0]
        
        bbs_switch_bb1_bb2 = sub_id * (nb_busbar * (nb_busbar - 1) // 2)  # switch between busbar 1 and busbar 2 at this substation
        load_id_switch = dtd.get_switch_id_ieee(dtd.load_to_conn_node_id[load_this[0]])
        gen_id_switch = dtd.get_switch_id_ieee(dtd.gen_to_conn_node_id[gen_this[0]])
        lor_id_switch = dtd.get_switch_id_ieee(dtd.line_or_to_conn_node_id[line_or_this[0]])
        lex_id_switch = dtd.get_switch_id_ieee(dtd.line_ex_to_conn_node_id[line_ex_this[0]])
        
        el_id_switch = load_id_switch
        el_this = load_this
        vect_topo_vect = type(self.env).load_pos_topo_vect
        for el_id_switch, el_this, vect_topo_vect, tag in zip([load_id_switch, gen_id_switch, lor_id_switch, lex_id_switch],
                                                              [load_this, gen_this, line_or_this, line_ex_this],
                                                              [type(self.env).load_pos_topo_vect, 
                                                               type(self.env).gen_pos_topo_vect,
                                                               type(self.env).line_or_pos_topo_vect,
                                                               type(self.env).line_ex_pos_topo_vect],
                                                              ["load", "gen", "lor", "lex"]):
            nb_when_disco = 58  # number of unaffected element
            if tag == "lor" or tag == "lex":
                # the other extremity is impacted in case I disconnect a line
                nb_when_disco = 57
            # disconnect the load with the breaker
            act = self.env.action_space({"set_switch": [(el_id_switch, -1)]})
            bk_act = self.env.backend.my_bk_act_class()
            bk_act += act
            topo_vect = bk_act()[2].values
            assert topo_vect[vect_topo_vect[el_this]] == -1, f"error for {tag}"
            assert (topo_vect == 1).sum() == nb_when_disco, f"error for {tag} : {(topo_vect == 1).sum()} vs {nb_when_disco}"
            
            # disconnect the load by disconnecting it of all the busbars
            act = self.env.action_space({"set_switch": [(el, -1) 
                                                        for el in range(el_id_switch + 1, el_id_switch + nb_busbar +1)]
                                         })
            bk_act = self.env.backend.my_bk_act_class()
            bk_act += act
            topo_vect = bk_act()[2].values
            assert topo_vect[vect_topo_vect[el_this]] == -1, f"error for {tag}"
            assert (topo_vect == 1).sum() == nb_when_disco, f"error for {tag} : {(topo_vect == 1).sum()} vs {nb_when_disco}"
            
            # now connect the load to busbar 2
            act = self.env.action_space({"set_switch": [(el, -1 if el != el_id_switch + 2 else 1) 
                                                        for el in range(el_id_switch + 1, el_id_switch + nb_busbar +1)]
                                         })
            bk_act = self.env.backend.my_bk_act_class()
            bk_act += act
            topo_vect = bk_act()[2].values
            assert topo_vect[vect_topo_vect[el_this]] == 2, f"error for {tag}"
            assert (topo_vect == 1).sum() == 58, f"error for {tag} : {(topo_vect == 1).sum()} vs 58"
            
            # load still on busbar 2, but disconnected
            act = self.env.action_space({"set_switch": ([(el, -1 if el != el_id_switch + 2 else 1) 
                                                        for el in range(el_id_switch + 1, el_id_switch + nb_busbar +1)]+ 
                                                        [(el_id_switch, -1)])
                                         })
            bk_act = self.env.backend.my_bk_act_class()
            bk_act += act
            topo_vect = bk_act()[2].values
            assert topo_vect[vect_topo_vect[el_this]] == -1, f"error for {tag}"
            assert (topo_vect == 1).sum() == nb_when_disco, f"error for {tag} : {(topo_vect == 1).sum()} vs {nb_when_disco}"
            
            # load on busbar 2, but busbars connected
            act = self.env.action_space({"set_switch": ([(el, -1 if el != el_id_switch + 2 else 1) 
                                                        for el in range(el_id_switch + 1, el_id_switch + nb_busbar +1)] +
                                                        [(bbs_switch_bb1_bb2, 1)])
                                         })
            bk_act = self.env.backend.my_bk_act_class()
            bk_act += act
            topo_vect = bk_act()[2].values
            assert topo_vect[vect_topo_vect[el_this]] == 1, f"error for {tag}"
            assert (topo_vect == 1).sum() == 59, f"error for {tag} : {(topo_vect == 1).sum()} vs 59"       
        
    # TODO detailed topo test print
    # TODO detailed topo test to_dict
    # TODO detailed topo test as_serializable_dict
    # TODO detailed topo test from_dict
    # TODO detailed topo test from_json (? does it exists)
    
    # then detailed topo 
    # TODO detailed topo test from_switches_position when there is a mask in the substation
    # TODO detailed topo test env.step only switch
    # TODO detailed topo test env.step switch and set_bus

class TestActAndBkAct(unittest.TestCase):
    def _aux_n_bb_per_sub(self):
        return 2
    
    def setUp(self) -> None:
        DetailedTopoTester.setUp(self)
        
    def test_no_switch_action(self):
        # TODO detailed topo: speed it up by
        # not using the routine to/ from switch !
        obs = self.env.reset()
        
        # agent does that
        act : BaseAction = self.env.action_space()
        
        # env does that
        bk_act : _BackendAction = self.env._backend_action
        bk_act += act
        
        # backend does that
        (
            active_bus, # TODO detailed topo does not compute that
            (prod_p, prod_v, load_p, load_q, storage),
            topo__,  # TODO detailed topo does not compute that
            shunts__,  # TODO detailed topo does not compute shunt_bus there
        ) = bk_act()
        switch_pos = bk_act.get_all_switches()
        tgt_switch = np.array([False, False, False, False, False, False, False, False, False,
                               False, False, False, False, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False])
        assert (switch_pos == tgt_switch).all()
        
        # env does that again
        bk_act.reset()
        
    def test_switch_bbc_action(self):    
        """do an action of closing a switch between busbars
        (which has no impact on topo)"""
        # TODO detailed topo: speed it up by
        # not using the routine to/ from switch !
        obs = self.env.reset()
        
        # agent does that
        act : BaseAction = self.env.action_space({"set_switch": [(0, 1)]})
        
        # env does that
        bk_act : _BackendAction = self.env._backend_action
        bk_act += act
        
        # backend does that
        (
            active_bus, # TODO detailed topo does not compute that
            (prod_p, prod_v, load_p, load_q, storage),
            topo__,  # TODO detailed topo does not compute that
            shunts__,  # TODO detailed topo does not compute shunt_bus there
        ) = bk_act()
        switch_pos = bk_act.get_all_switches()
        tgt_switch = np.array([True, False, False, False, False, False, False, False, False,
                               False, False, False, False, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False,  True,  True, False,  True,
                                True, False,  True,  True, False])
        assert (switch_pos == tgt_switch).all()
        
        # env does that again
        bk_act.reset()
        
    
# TODO detailed topo test no shunt too
# TODO detailed topo test "_get_full_cls_str"(experimental_read_from_local_dir)
# TODO detailed topo test with different n_busbar_per_sub
# TODO detailed topo test action
# TODO detailed topo test observation
# TODO detailed topo test agent that do both actions on switches and with set_bus / change_bus
# TODO detailed topo test agent that act on switches but with an opponent that disconnect lines
 
if __name__ == "__main__":
    unittest.main()
   