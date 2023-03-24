# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Chronics.handlers.baseHandler import BaseHandler


class DoNothingHandler(BaseHandler):
    """This is the type of Time Series Handler that does nothing. 
    
    The environment will act as if the time series this Handler is reponsible for will never change.
    """
    def __init__(self, array_name="do nothing") -> None:
        super().__init__("do nothing")
    
    def initialize(self, order_backend_prods, names_chronics_to_backend):
        pass
    
    def check_validity(self, backend):
        pass
    
    def load_next(self, dict):
        pass
        
    def done(self):
        return False
        
    def forecast(self,
                 forecast_horizon_id,
                 inj_dict_env,
                 inj_dict_previous_forecast,
                 # eg gen_p_handler if this is set to gen_p_for_handler:
                 env_handler,  
                 # list of the 4 env handlers: (load_p_handler, load_q_handler, gen_p_handler, gen_v_handler)
                 env_handlers):
        return None
    
    def load_next_maintenance(self):
        return None, None
    
    def load_next_hazard(self):
        return None
    