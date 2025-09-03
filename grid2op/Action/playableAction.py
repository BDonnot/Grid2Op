# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
from typing import Optional, Dict, Literal

from grid2op.Exceptions import AmbiguousAction
from grid2op.Action.baseAction import BaseAction
from grid2op.Exceptions.grid2OpException import Grid2OpException


class PlayableAction(BaseAction):
    """
    From this class inherit all actions that the player will be allowed to do. This includes for example
    :class:`TopologyAndDispatchAction` or :class:`TopologyAction`
    """

    authorized_keys = {
        "set_line_status",
        "change_line_status",
        "set_bus",
        "change_bus",
        "redispatch",
        "set_storage",
        "curtail",
        "raise_alarm",
        "raise_alert",
        "detach_load",  # new in 1.11.0
        "detach_gen",  # new in 1.11.0
        "detach_storage",  # new in 1.11.0
    }

    attr_list_vect = [
        "_set_line_status",
        "_switch_line_status",
        "_set_topo_vect",
        "_change_bus_vect",
        "_redispatch",
        "_storage_power",
        "_curtail",
        "_raise_alarm",
        "_raise_alert",
        "_detach_load",  # new in 1.11.0
        "_detach_gen",  # new in 1.11.0
        "_detach_storage",  # new in 1.11.0
    ]
    attr_list_set = set(attr_list_vect)
    shunt_added = True  # no shunt here

    authorized_keys_to_digest = {
        "set_line_status": BaseAction._digest_set_status,
        "change_line_status": BaseAction._digest_change_status,
        "set_bus": BaseAction._digest_setbus,
        "change_bus": BaseAction._digest_change_bus,
        "redispatch": BaseAction._digest_redispatching,
        "set_storage": BaseAction._digest_storage,
        "curtail": BaseAction._digest_curtailment,
        "raise_alarm": BaseAction._digest_alarm,
        "raise_alert": BaseAction._digest_alert,
        "detach_load": BaseAction._digest_detach_load,  # new in 1.11.0
        "detach_gen": BaseAction._digest_detach_gen,  # new in 1.11.0
        "detach_storage": BaseAction._digest_detach_storage,  # new in 1.11.0
        "set_switch": BaseAction._digest_set_switch,
        "change_switch": BaseAction._digest_change_switch,
    }
    
    def __init__(self, _names_chronics_to_backend: Optional[Dict[Literal["loads", "prods", "lines"], Dict[str, str]]]=None):
        super().__init__(_names_chronics_to_backend)
        
    def __call__(self):
        """
         .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Compare to the ancestor :func:`BaseAction.__call__` this type of BaseAction doesn't allow internal actions
        The returned tuple is same, but with empty dictionaries for internal actions

        Returns
        -------
        dict_injection: ``dict``
            This dictionary is always empty

        set_line_status: :class:`numpy.ndarray`, dtype:int
            This array is :attr:`BaseAction._set_line_status`

        switch_line_status: :class:`numpy.ndarray`, dtype:bool
            This array is :attr:`BaseAction._switch_line_status`

        set_topo_vect: :class:`numpy.ndarray`, dtype:int
            This array is :attr:`BaseAction._set_topo_vect`

        change_bus_vect: :class:`numpy.ndarray`, dtype:bool
            This array is :attr:`BaseAction._change_bus_vect`

        redispatch: :class:`numpy.ndarray`, dtype:float
            The array is :attr:`BaseAction._redispatch`

        curtail: :class:`numpy.ndarray`, dtype:float
            The array is :attr:`BaseAction._curtail`

        shunts: ``dict``
            Always empty for this class
        """
        if self._dict_inj:
            raise AmbiguousAction("Injections actions are not playable.")

        self._check_for_ambiguity()
        return (
            {},
            self._set_line_status,
            self._switch_line_status,
            self._set_topo_vect,
            self._change_bus_vect,
            self._redispatch,
            self._storage_power,
            {},
        )

    @classmethod
    def _add_shunt_data(cls):
        # don't add shunts for this class
        pass
    
    @classmethod
    def _aux_act_derived_finalize_class_definition(cls):
        # added check to make sure the class
        # is well formed
        for el in cls.attr_list_vect:
            mapping_human = cls.mapping_vect_auth_keys[el]
            if (mapping_human in cls.authorized_keys and 
                mapping_human not in cls.authorized_keys_to_digest):
                raise Grid2OpException(f"Misformed action class: attribute {el} "
                                        f"(usable with key {mapping_human}) should "
                                        "be in cls.authorized_keys_to_digest")
    
    def update(self, dict_):
        """
         .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Similar to :class:`BaseAction`, except that the allowed entries are limited to the playable action set

        Parameters
        ----------
        dict_: :class:`dict`
            See the help of :func:`BaseAction.update` for a detailed explanation. 
            If an entry is not in the playable action set, this will raise

        Returns
        -------
        self: :class:`PlayableAction`
            Return object itself thus allowing multiple calls to "update" to be chained.

        """

        self._reset_vect()
        warn_msg = (
            'The key "{}" used to update an action will be ignored. Valid keys are {}'
        )

        if dict_ is None:
            return self
        cls = type(self)
        for kk in dict_.keys():
            if kk not in cls.authorized_keys:
                warn = warn_msg.format(kk, cls.authorized_keys)
                warnings.warn(warn)
            else:
                cls.authorized_keys_to_digest[kk](self, dict_)

        return self
