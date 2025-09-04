# Copyright (c) 2025, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Space import AddDetailedTopoIEEE
from grid2op.Backend.pandaPowerBackend import PandaPowerBackend  # or any other backend (*eg* lightsim2grid)

class PandaPowerBackendWithDetailedTopoIEEE(AddDetailedTopoIEEE, PandaPowerBackend):
    pass
