# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings

import grid2op
from grid2op.Parameters import Parameters
from grid2op.dtypes import dt_float
from grid2op.Action import PlayableAction
from grid2op.Backend import PandaPowerBackendWithDetailedTopoIEEE

# TODO check when there is also redispatching


class TestDecompUnary(unittest.TestCase):
    """test the env part of the storage functionality"""

    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            param = Parameters()
            param.NO_OVERFLOW_DISCONNECTION = True
            self.env = grid2op.make(
                "educ_case14_storage",
                backend=PandaPowerBackendWithDetailedTopoIEEE(),
                test=True,
                allow_detachment=True,
                action_class=PlayableAction,
                _add_to_name=type(self).__name__,
            )
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_change_bus(self):
        act = self.env.action_space({"change_bus": {"loads_id": [0, 1],
                                                    "generators_id": [0]}})
        res = act.decompose_as_unary_actions()
        assert len(res) == 1
        assert "change_bus" in res
        assert len(res["change_bus"]) == 2
        
        assert res["change_bus"][0]._change_bus_vect[act.load_pos_topo_vect[0]]
        assert res["change_bus"][0]._change_bus_vect[act.gen_pos_topo_vect[0]]
        assert not res["change_bus"][0]._change_bus_vect[act.load_pos_topo_vect[1]]
        
        assert not res["change_bus"][1]._change_bus_vect[act.load_pos_topo_vect[0]]
        assert not res["change_bus"][1]._change_bus_vect[act.gen_pos_topo_vect[0]]
        assert res["change_bus"][1]._change_bus_vect[act.load_pos_topo_vect[1]]
        
        res = act.decompose_as_unary_actions(group_topo=True)
        assert len(res) == 1
        assert "change_bus" in res
        assert len(res["change_bus"]) == 1
        assert res["change_bus"][0]._change_bus_vect[act.load_pos_topo_vect[0]]
        assert res["change_bus"][0]._change_bus_vect[act.gen_pos_topo_vect[0]]
        assert res["change_bus"][0]._change_bus_vect[act.load_pos_topo_vect[1]]
        
    def test_set_bus(self):
        act = self.env.action_space({"set_bus": {"loads_id": [(0, 2), (1, 2)],
                                                 "generators_id": [(0, 2)]}})
        res = act.decompose_as_unary_actions()
        assert len(res) == 1
        assert "set_bus" in res
        assert len(res["set_bus"]) == 2
        assert res["set_bus"][0]._set_topo_vect[act.load_pos_topo_vect[0]] == 2
        assert res["set_bus"][0]._set_topo_vect[act.gen_pos_topo_vect[0]] == 2
        assert res["set_bus"][0]._set_topo_vect[act.load_pos_topo_vect[1]] == 0
        assert res["set_bus"][1]._set_topo_vect[act.load_pos_topo_vect[0]] == 0
        assert res["set_bus"][1]._set_topo_vect[act.gen_pos_topo_vect[0]] == 0
        assert res["set_bus"][1]._set_topo_vect[act.load_pos_topo_vect[1]] == 2
        
        res = act.decompose_as_unary_actions(group_topo=True)
        assert len(res) == 1
        assert "set_bus" in res
        assert len(res["set_bus"]) == 1
        assert res["set_bus"][0]._set_topo_vect[act.load_pos_topo_vect[0]] == 2
        assert res["set_bus"][0]._set_topo_vect[act.gen_pos_topo_vect[0]] == 2
        assert res["set_bus"][0]._set_topo_vect[act.load_pos_topo_vect[1]] == 2
        
    def test_set_ls(self):
        act = self.env.action_space({"set_line_status": [(0, -1), (1, -1)]})
        res = act.decompose_as_unary_actions()
        assert len(res) == 1
        assert "set_line_status" in res
        assert len(res["set_line_status"]) == 2
        assert res["set_line_status"][0]._set_line_status[0] == -1
        assert not res["set_line_status"][0]._set_line_status[1] == -1
        assert not res["set_line_status"][1]._set_line_status[0] == -1
        assert res["set_line_status"][1]._set_line_status[1] == -1
        
        res = act.decompose_as_unary_actions(group_line_status=True)
        assert len(res) == 1
        assert "set_line_status" in res
        assert len(res["set_line_status"]) == 1
        assert res["set_line_status"][0]._set_line_status[0] == -1
        assert res["set_line_status"][0]._set_line_status[1] == -1
        
    def test_change_ls(self):
        act = self.env.action_space({"change_line_status": [0, 1]})
        res = act.decompose_as_unary_actions()
        assert len(res) == 1
        assert "change_line_status" in res
        assert len(res["change_line_status"]) == 2
        assert res["change_line_status"][0]._switch_line_status[0]
        assert not res["change_line_status"][0]._switch_line_status[1]
        assert not res["change_line_status"][1]._switch_line_status[0]
        assert res["change_line_status"][1]._switch_line_status[1]
        
        res = act.decompose_as_unary_actions(group_line_status=True)
        assert len(res) == 1
        assert "change_line_status" in res
        assert len(res["change_line_status"]) == 1
        assert res["change_line_status"][0]._switch_line_status[0]
        assert res["change_line_status"][0]._switch_line_status[1]
        
    def test_redispatch(self):
        act = self.env.action_space({"redispatch": [(0, +1.), (1, -1.)]})
        res = act.decompose_as_unary_actions(group_redispatch=False)
        assert len(res) == 1
        assert "redispatch" in res
        assert len(res["redispatch"]) == 2
        assert res["redispatch"][0]._redispatch[0] == 1.
        assert res["redispatch"][0]._redispatch[1] == 0.
        assert res["redispatch"][1]._redispatch[0] == 0.
        assert res["redispatch"][1]._redispatch[1] == -1.
        
        res = act.decompose_as_unary_actions(group_redispatch=True)
        assert len(res) == 1
        assert "redispatch" in res
        assert len(res["redispatch"]) == 1
        assert res["redispatch"][0]._redispatch[0] == 1.
        assert res["redispatch"][0]._redispatch[1] == -1.
        
    def test_storage(self):
        act = self.env.action_space({"set_storage": [(0, +1.), (1, -1.)]})
        res = act.decompose_as_unary_actions(group_storage=False)
        assert len(res) == 1
        assert "set_storage" in res
        assert len(res["set_storage"]) == 2
        assert res["set_storage"][0]._storage_power[0] == 1.
        assert res["set_storage"][0]._storage_power[1] == 0.
        assert res["set_storage"][1]._storage_power[0] == 0.
        assert res["set_storage"][1]._storage_power[1] == -1.
        
        res = act.decompose_as_unary_actions(group_storage=True)
        assert len(res) == 1
        assert "set_storage" in res
        assert len(res["set_storage"]) == 1
        assert res["set_storage"][0]._storage_power[0] == 1.
        assert res["set_storage"][0]._storage_power[1] == -1.
        
    def test_curtail(self):
        act = self.env.action_space({"curtail": [(4, 0.8), (5, 0.7)]})
        res = act.decompose_as_unary_actions(group_curtail=False)
        assert len(res) == 1
        assert "curtail" in res
        assert len(res["curtail"]) == 2
        assert res["curtail"][0]._curtail[4] == dt_float(0.8)
        assert res["curtail"][0]._curtail[5] == -1.0
        assert res["curtail"][1]._curtail[4] == -1.0
        assert res["curtail"][1]._curtail[5] == dt_float(0.7)
        
        res = act.decompose_as_unary_actions(group_curtail=True)
        assert len(res) == 1
        assert "curtail" in res
        assert len(res["curtail"]) == 1
        assert res["curtail"][0]._curtail[4] == dt_float(0.8)
        assert res["curtail"][0]._curtail[5] == dt_float(0.7)
    
    def test_detachment(self):
        for attr_nm in ['gen', "load", "storage"]:
            flag = f"detach_{attr_nm}"
            act = self.env.action_space({flag: [0, 1]})
            res = act.decompose_as_unary_actions(group_detach=False)
            assert len(res) == 1
            assert flag in res
            assert len(res[flag]) == 2
            assert getattr(res[flag][0], flag)[0]
            assert not getattr(res[flag][0], flag)[1]
            assert not getattr(res[flag][1], flag)[0]
            assert getattr(res[flag][1], flag)[1]
            
            res = act.decompose_as_unary_actions(group_detach=True)
            assert len(res) == 1
            assert flag in res
            assert len(res[flag]) == 1
            assert getattr(res[flag][0], flag)[0]
            assert getattr(res[flag][0], flag)[1]
    
    def test_set_switches(self):
        # switch 0 and 62 are on substation 0
        # switch 1 on substation 1
        act = self.env.action_space({"set_switch": [(0, 1), (1, 1), (62, -1)]})
        res = act.decompose_as_unary_actions(group_switch=False)
        assert len(res) == 1
        assert "set_switch" in res
        assert len(res["set_switch"]) == 2, f'{len(res["set_switch"])} vs 2'
        assert res["set_switch"][0]._set_switch_status[0] == 1
        assert res["set_switch"][0]._set_switch_status[1] == 0
        assert res["set_switch"][0]._set_switch_status[62] == -1
        assert res["set_switch"][1]._set_switch_status[0] == 0
        assert res["set_switch"][1]._set_switch_status[1] == 1
        assert res["set_switch"][1]._set_switch_status[62] == 0
        
        res = act.decompose_as_unary_actions(group_switch=True)
        assert len(res) == 1
        assert "set_switch" in res
        assert len(res["set_switch"]) == 1, f'{len(res["set_switch"])} vs 1'
        assert res["set_switch"][0]._set_switch_status[0] == 1
        assert res["set_switch"][0]._set_switch_status[1] == 1
        assert res["set_switch"][0]._set_switch_status[62] == -1
        
    def test_change_switches(self):
        # switch 0 and 62 are on substation 0
        # switch 1 on substation 1
        act = self.env.action_space({"change_switch": [0, 1, 62]})
        res = act.decompose_as_unary_actions(group_switch=False)
        assert len(res) == 1
        assert "change_switch" in res
        assert len(res["change_switch"]) == 2, f'{len(res["change_switch"])} vs 2'
        assert res["change_switch"][0]._change_switch_status[0]
        assert not res["change_switch"][0]._change_switch_status[1]
        assert res["change_switch"][0]._change_switch_status[62]
        assert not res["change_switch"][1]._change_switch_status[0]
        assert res["change_switch"][1]._change_switch_status[1]
        assert not res["change_switch"][1]._change_switch_status[62]
        
        res = act.decompose_as_unary_actions(group_switch=True)
        assert len(res) == 1
        assert "change_switch" in res
        assert len(res["change_switch"]) == 1, f'{len(res["change_switch"])} vs 1'
        assert res["change_switch"][0]._change_switch_status[0]
        assert res["change_switch"][0]._change_switch_status[1]
        assert res["change_switch"][0]._change_switch_status[62]
        
    def test_all(self):
        act = self.env.action_space({"curtail": [(4, 0.8), (5, 0.7)],
                                     "set_storage": [(0, +1.), (1, -1.)],
                                     "redispatch": [(0, +1.), (1, -1.)],
                                     "change_line_status": [2, 3],
                                     "set_line_status": [(0, -1), (1, -1)],
                                     "set_bus": {"loads_id": [(0, 2), (1, 2)],
                                                 "generators_id": [(0, 2)]},
                                     "change_bus": {"loads_id": [2, 3],
                                                    "generators_id": [1]}
                                     })
        res = act.decompose_as_unary_actions()
        assert len(res) == 7
        assert "curtail" in res
        assert "curtail" in res
        assert "redispatch" in res
        assert "change_line_status" in res
        assert "set_line_status" in res
        assert "set_bus" in res
        assert "change_bus" in res
        tmp = self.env.action_space()
        for k, v in res.items():
            for a in v:
                tmp += a
        assert tmp == act
        
if __name__ == "__main__":
    unittest.main()
