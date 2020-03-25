"""
This file contains the settings (path to the case file, chronics converter etc. that allows to run
the competition "L2RPN 2019" that took place on the pypownet plateform.

It is present to reproduce this competition.
"""
import os
import pkg_resources
import copy
import warnings

from datetime import timedelta, datetime
import numpy as np
import pandas as pd

try:
    from .ChronicsHandler import GridStateFromFileWithForecasts
    from .Action import Action, AmbiguousAction, IncorrectNumberOfElements
except (ModuleNotFoundError, ImportError):
    from ChronicsHandler import GridStateFromFileWithForecasts
    from Action import Action, AmbiguousAction, IncorrectNumberOfElements

# the reference powergrid was different than the default case14 of the litterature.
L2RPN2019_CASEFILE = os.path.abspath(os.path.join(pkg_resources.resource_filename(__name__, "data"),
                                            "test_PandaPower", "L2RPN_2019_grid.json"))

CASE_14_L2RPN2019_LAYOUT = graph_layout = [(-280, -81), (-100, -270), (366, -270), (366, -54), (-64, -54), (-64, 54),
                                           (450, 0), (550, 0), (326, 54), (222, 108), (79, 162), (-170, 270),
                                           (-64, 270), (222, 216)]

# names of object of the grid were not in the same order as the default one
L2RPN2019_DICT_NAMES = {'loads': {'2_C-10.61': 'load_1_0',
                                       '3_C151.15': 'load_2_1',
                                       '14_C63.6': 'load_13_10',
                                       '4_C-9.47': 'load_3_2',
                                       '5_C201.84': 'load_4_3',
                                       '6_C-6.27': 'load_5_4',
                                       '9_C130.49': 'load_8_5',
                                       '10_C228.66': 'load_9_6',
                                       '11_C-138.89': 'load_10_7',
                                       '12_C-27.88': 'load_11_8',
                                       '13_C-13.33': 'load_12_9'},
                             'lines': {'1_2_1': '0_1_0',
                                       '1_5_2': '0_4_1',
                                       '9_10_16': '8_9_16',
                                       '9_14_17': '8_13_15',
                                       '10_11_18': '9_10_17',
                                       '12_13_19': '11_12_18',
                                       '13_14_20': '12_13_19',
                                       '2_3_3': '1_2_2',
                                       '2_4_4': '1_3_3',
                                       '2_5_5': '1_4_4',
                                       '3_4_6': '2_3_5',
                                       '4_5_7': '3_4_6',
                                       '6_11_11': '5_10_12',
                                       '6_12_12': '5_11_11',
                                       '6_13_13': '5_12_10',
                                       '4_7_8': '3_6_7',
                                       '4_9_9': '3_8_8',
                                       '5_6_10': '4_5_9',
                                       '7_8_14': '6_7_13',
                                       '7_9_15': '6_8_14'},
                             'prods': {'1_G137.1': 'gen_0_4',
                                       '3_G36.31': 'gen_1_0',
                                       '6_G63.29': 'gen_2_1',
                                       '2_G-56.47': 'gen_5_2',
                                       '8_G40.43': 'gen_7_3'}}


# Names of the csv were not the same
class ReadPypowNetData(GridStateFromFileWithForecasts):
    def __init__(self, path, sep=";", time_interval=timedelta(minutes=5),
                 max_iter=-1,
                 chunk_size=None):
        GridStateFromFileWithForecasts.__init__(self, path, sep=sep, time_interval=time_interval,
                                                max_iter=max_iter, chunk_size=chunk_size)

    def initialize(self, order_backend_loads, order_backend_prods, order_backend_lines, order_backend_subs,
                   names_chronics_to_backend=None):
        """
        TODO Doc
        """
        self.n_gen = len(order_backend_prods)
        self.n_load = len(order_backend_loads)
        self.n_line = len(order_backend_lines)

        self.names_chronics_to_backend = copy.deepcopy(names_chronics_to_backend)
        if self.names_chronics_to_backend is None:
            self.names_chronics_to_backend = {}
        if not "loads" in self.names_chronics_to_backend:
            self.names_chronics_to_backend["loads"] = {k: k for k in order_backend_loads}
        else:
            self._assert_correct(self.names_chronics_to_backend["loads"], order_backend_loads)
        if not "prods" in self.names_chronics_to_backend:
            self.names_chronics_to_backend["prods"] = {k: k for k in order_backend_prods}
        else:
            self._assert_correct(self.names_chronics_to_backend["prods"], order_backend_prods)
        if not "lines" in self.names_chronics_to_backend:
            self.names_chronics_to_backend["lines"] = {k: k for k in order_backend_lines}
        else:
            self._assert_correct(self.names_chronics_to_backend["lines"], order_backend_lines)
        if not "subs" in self.names_chronics_to_backend:
            self.names_chronics_to_backend["subs"] = {k: k for k in order_backend_subs}
        else:
            self._assert_correct(self.names_chronics_to_backend["subs"], order_backend_subs)

        # print(os.listdir(self.path))
        read_compressed = ".csv"
        if not os.path.exists(os.path.join(self.path, "_N_loads_p.csv")):
            # try to read compressed data
            if os.path.exists(os.path.join(self.path, "_N_loads_p.csv.bz2")):
                read_compressed = ".csv.bz2"
            elif os.path.exists(os.path.join(self.path, "_N_loads_p.zip")):
                read_compressed = ".zip"
            elif os.path.exists(os.path.join(self.path, "_N_loads_p.csv.gzip")):
                read_compressed = ".csv.gzip"
            elif os.path.exists(os.path.join(self.path, "_N_loads_p.csv.xz")):
                read_compressed = ".csv.xz"
            else:
                raise RuntimeError(
                    "GridStateFromFile: unable to locate the data files that should be at \"{}\"".format(self.path))
        load_p = pd.read_csv(os.path.join(self.path, "_N_loads_p{}".format(read_compressed)), sep=self.sep)
        load_q = pd.read_csv(os.path.join(self.path, "_N_loads_q{}".format(read_compressed)), sep=self.sep)
        prod_p = pd.read_csv(os.path.join(self.path, "_N_prods_p{}".format(read_compressed)), sep=self.sep)
        prod_v = pd.read_csv(os.path.join(self.path, "_N_prods_v{}".format(read_compressed)), sep=self.sep)
        hazards = pd.read_csv(os.path.join(self.path, "hazards{}".format(read_compressed)), sep=self.sep)
        maintenance = pd.read_csv(os.path.join(self.path, "maintenance{}".format(read_compressed)), sep=self.sep)

        order_backend_loads = {el: i for i, el in enumerate(order_backend_loads)}
        order_backend_prods = {el: i for i, el in enumerate(order_backend_prods)}
        order_backend_lines = {el: i for i, el in enumerate(order_backend_lines)}

        order_chronics_load_p = np.array([order_backend_loads[self.names_chronics_to_backend["loads"][el]]
                                          for el in load_p.columns]).astype(np.int)
        order_backend_load_q = np.array([order_backend_loads[self.names_chronics_to_backend["loads"][el]]
                                         for el in load_q.columns]).astype(np.int)
        order_backend_prod_p = np.array([order_backend_prods[self.names_chronics_to_backend["prods"][el]]
                                         for el in prod_p.columns]).astype(np.int)
        order_backend_prod_v = np.array([order_backend_prods[self.names_chronics_to_backend["prods"][el]]
                                         for el in prod_v.columns]).astype(np.int)
        order_backend_hazards = np.array([order_backend_lines[self.names_chronics_to_backend["lines"][el]]
                                          for el in hazards.columns]).astype(np.int)
        order_backend_maintenance = np.array([order_backend_lines[self.names_chronics_to_backend["lines"][el]]
                                              for el in maintenance.columns]).astype(np.int)

        self.load_p = copy.deepcopy(load_p.values[:, np.argsort(order_chronics_load_p)])
        self.load_q = copy.deepcopy(load_q.values[:, np.argsort(order_backend_load_q)])
        self.prod_p = copy.deepcopy(prod_p.values[:, np.argsort(order_backend_prod_p)])
        self.prod_v = copy.deepcopy(prod_v.values[:, np.argsort(order_backend_prod_v)])
        self.hazards = copy.deepcopy(hazards.values[:, np.argsort(order_backend_hazards)])
        self.maintenance = copy.deepcopy(maintenance.values[:, np.argsort(order_backend_maintenance)])

        # date and time
        datetimes_ = pd.read_csv(os.path.join(self.path, "_N_datetimes{}".format(read_compressed)), sep=self.sep)
        self.start_datetime = datetime.strptime(datetimes_.iloc[0, 0], "%Y-%b-%d")

        # there are maintenance and hazards only if the value in the file is not 0.
        self.maintenance = self.maintenance != 0.
        self.hazards = self.hazards != 0.

        self.curr_iter = 0
        if self.max_iter == -1:
            # if the number of maximum time step is not set yet, we set it to be the number of
            # data in the chronics (number of rows of the files) -1.
            # the -1 is present because the initial grid state doesn't count as a "time step" but is read
            # from these data.
            self.max_iter = self.load_p.shape[0]-1

        load_p = pd.read_csv(os.path.join(self.path, "_N_loads_p_planned{}".format(read_compressed)), sep=self.sep)
        load_q = pd.read_csv(os.path.join(self.path, "_N_loads_q_planned{}".format(read_compressed)), sep=self.sep)
        prod_p = pd.read_csv(os.path.join(self.path, "_N_prods_p_planned{}".format(read_compressed)), sep=self.sep)
        prod_v = pd.read_csv(os.path.join(self.path, "_N_prods_v_planned{}".format(read_compressed)), sep=self.sep)
        maintenance = pd.read_csv(os.path.join(self.path, "maintenance{}".format(read_compressed)),
                                  sep=self.sep)

        order_backend_loads = {el: i for i, el in enumerate(order_backend_loads)}
        order_backend_prods = {el: i for i, el in enumerate(order_backend_prods)}
        order_backend_lines = {el: i for i, el in enumerate(order_backend_lines)}

        order_chronics_load_p = np.array([order_backend_loads[self.names_chronics_to_backend["loads"][el]]
                                          for el in load_p.columns]).astype(np.int)
        order_backend_load_q = np.array([order_backend_loads[self.names_chronics_to_backend["loads"][el]]
                                         for el in load_q.columns]).astype(np.int)
        order_backend_prod_p = np.array([order_backend_prods[self.names_chronics_to_backend["prods"][el]]
                                         for el in prod_p.columns]).astype(np.int)
        order_backend_prod_v = np.array([order_backend_prods[self.names_chronics_to_backend["prods"][el]]
                                         for el in prod_v.columns]).astype(np.int)
        order_backend_maintenance = np.array([order_backend_lines[self.names_chronics_to_backend["lines"][el]]
                                              for el in maintenance.columns]).astype(np.int)

        self.load_p_forecast = copy.deepcopy(load_p.values[:, np.argsort(order_chronics_load_p)])
        self.load_q_forecast = copy.deepcopy(load_q.values[:, np.argsort(order_backend_load_q)])
        self.prod_p_forecast = copy.deepcopy(prod_p.values[:, np.argsort(order_backend_prod_p)])
        self.prod_v_forecast = copy.deepcopy(prod_v.values[:, np.argsort(order_backend_prod_v)])
        self.maintenance_forecast = copy.deepcopy(maintenance.values[:, np.argsort(order_backend_maintenance)])

        # there are maintenance and hazards only if the value in the file is not 0.
        self.maintenance_time = np.zeros(shape=(self.load_p.shape[0], self.n_line), dtype=np.int) - 1
        self.maintenance_duration = np.zeros(shape=(self.load_p.shape[0], self.n_line), dtype=np.int)
        self.hazard_duration = np.zeros(shape=(self.load_p.shape[0], self.n_line), dtype=np.int)
        for line_id in range(self.n_line):
            self.maintenance_time[:, line_id] = self.get_maintenance_time_1d(self.maintenance[:, line_id])
            self.maintenance_duration[:, line_id] = self.get_maintenance_duration_1d(self.maintenance[:, line_id])
            self.hazard_duration[:, line_id] = self.get_maintenance_duration_1d(self.hazards[:, line_id])

        self.maintenance_forecast = self.maintenance != 0.

        self.curr_iter = 0
        if self.maintenance is not None:
            n_ = self.maintenance.shape[0]
        elif self.hazards is not None:
            n_ = self.hazards.shape[0]
        else:
            n_ = None
            for fn in ["prod_p", "load_p", "prod_v", "load_q"]:
                ext_ = self._get_fileext(fn)
                if ext_ is not None:
                    n_ = self._file_len(os.path.join(self.path, "{}{}".format(fn, ext_)), ext_)
                    break
            if n_ is None:
                raise ChronicsError("No files are found in directory \"{}\". If you don't want to load any chronics,"
                                    " use  \"ChangeNothing\" and not \"{}\" to load chronics."
                                    "".format(self.path, type(self)))
        self.n_ = n_  # the -1 is present because the initial grid state doesn't count as a "time step"
        self.tmp_max_index = load_p.shape[0]

# class of the action didn't implement the "set" part. Only change was present.
# Beside when reconnected, objects were always reconnected on bus 1.
# This is not used at the moment.
class L2RPN2019_Action(Action):
    """
    This class is here to model only a subpart of Topological actions, the one consisting in topological switching.
    It will throw an "AmbiguousAction" error it someone attempt to change injections in any ways.

    It has the same attributes as its base class :class:`Action`.

    It is also here to show an example on how to implement a valid class deriving from :class:`Action`.

    **NB** This class doesn't allow to connect object to other buses than their original bus. In this case,
    reconnecting a powerline cannot be considered "ambiguous". We have to
    """
    def __init__(self, gridobj):
        """
        See the definition of :func:`Action.__init__` and of :class:`Action` for more information. Nothing more is done
        in this constructor.
        """
        Action.__init__(self, gridobj)

        # the injection keys is not authorized, meaning it will send a warning is someone try to implement some
        # modification injection.
        self.authorized_keys = set([k for k in self.authorized_keys
                                    if k != "injection" and k != "set_bus" and "set_line_status"])

    def __call__(self):
        """
        Compare to the ancestor :func:`Action.__call__` this type of Action doesn't allow to change the injections.
        The only difference is in the returned value *dict_injection* that is always an empty dictionnary.

        Returns
        -------
        dict_injection: :class:`dict`
            This dictionnary is always empty

        set_line_status: :class:`numpy.array`, dtype:int
            This array is :attr:`Action._set_line_status`

        switch_line_status: :class:`numpy.array`, dtype:bool
            This array is :attr:`Action._switch_line_status`, it is never modified

        set_topo_vect: :class:`numpy.array`, dtype:int
            This array is :attr:`Action._set_topo_vect`, it is never modified

        change_bus_vect: :class:`numpy.array`, dtype:bool
            This array is :attr:`Action._change_bus_vect`, it is never modified
        """
        if self._dict_inj:
            raise AmbiguousAction("You asked to modify the injection with an action of class \"TopologyAction\".")
        self._check_for_ambiguity()
        return {}, self._set_line_status, self._switch_line_status, self._set_topo_vect, self._change_bus_vect

    def update(self, dict_):
        """
        As its original implementation, this method allows to modify the way a dictionnary can be mapped to a valid
        :class:`Action`.

        It has only minor modifications compared to the original :func:`Action.update` implementation, most notably, it
        doesn't update the :attr:`Action._dict_inj`. It raises a warning if attempting to change them.

        Parameters
        ----------
        dict_: :class:`dict`
            See the help of :func:`Action.update` for a detailed explanation. **NB** all the explanations concerning the
            "injection", "change bus", "set bus", or "change line status" are irrelevant for this subclass.

        Returns
        -------
        self: :class:`PowerLineSet`
            Return object itself thus allowing mutiple call to "update" to be chained.
        """

        self.as_vect = None
        if dict_ is not None:
            for kk in dict_.keys():
                if kk not in self.authorized_keys:
                    warn = "The key \"{}\" used to update an action will be ignored. Valid keys are {}"
                    warn = warn.format(kk, self.authorized_keys)
                    warnings.warn(warn)

            self._digest_change_bus(dict_)
            self._digest_hazards(dict_)
            self._digest_maintenance(dict_)
            self._digest_change_status(dict_)

        # self.disambiguate_reconnection()

        return self

    def size(self):
        """
        Compare to the base class, this action has a shorter size, as all information about injections are ignored.
        Returns
        -------
        size: ``int``
            The size of :class:`PowerLineSet` converted to an array.
        """
        return self.n_line + self.dim_topo

    def to_vect(self):
        """
        See :func:`Action.to_vect` for a detailed description of this method.

        This method has the same behaviour as its base class, except it doesn't require any information about the
        injections to be sent, thus being more efficient from a memory footprint perspective.

        Returns
        -------
        _vectorized: :class:`numpy.array`, dtype:float
            The instance of this action converted to a vector.
        """
        if self.as_vect is None:
            self.as_vect = np.concatenate((
                self._switch_line_status.flatten().astype(np.float),
                self._change_bus_vect.flatten().astype(np.float)
                                           ))

            if self.as_vect.shape[0] != self.size():
                raise AmbiguousAction("L2RPN2019_Action has not the proper shape.")

        return self.as_vect

    def from_vect(self, vect):
        """
        See :func:`Action.from_vect` for a detailed description of this method.

        Nothing more is made except the initial vector is (much) smaller.

        Parameters
        ----------
        vect: :class:`numpy.array`, dtype:float
            A vector reprenseting an instance of :class:`.`

        Returns
        -------

        """
        self.reset()
        # pdb.set_trace()
        if vect.shape[0] != self.size():
            raise IncorrectNumberOfElements("Incorrect number of elements found while loading a \"TopologyAction\" from a vector. Found {} elements instead of {}".format(vect.shape[1], self.size()))
        prev_ = 0
        next_ = self.n_line

        self._switch_line_status = vect[prev_:next_]
        self._switch_line_status = self._switch_line_status.astype(np.bool); prev_=next_; next_+= self.dim_topo
        self._change_bus_vect = vect[prev_:next_]
        self._change_bus_vect = self._change_bus_vect.astype(np.bool)

        # self.disambiguate_reconnection()

        self._check_for_ambiguity()

    # def disambiguate_reconnection(self):
    #     """
    #     As this class doesn't allow to perform any topology change, when a powerline is reconnected, it's necessarily
    #     on the first bus of the substation.
    #
    #     So it's not ambiguous in this case. We have to implement this logic here, and that is what is done in this
    #     function.
    #
    #     """
    #     sel_ = self._set_line_status == 1
    #     if np.any(sel_):
    #         self._set_topo_vect[self.line_ex_pos_topo_vect[sel_]] = 1
    #         self._set_topo_vect[self.line_or_pos_topo_vect[sel_]] = 1

    def sample(self, space_prng):
        """
        Sample a PowerlineSwitch Action.

        By default, this sampling will act on one random powerline, and it will either
        disconnect it or reconnect it each with equal probability.

        Parameters
        ----------
        space_prng

        Returns
        -------
        res: :class:`PowerLineSwitch`
            The sampled action
        """
        self.reset()
        i = np.random.randint(0, self.size())  # the powerline on which we can act
        val = 2*np.random.randint(0, 2) - 1  # the action: +1 reconnect it, -1 disconnect it
        self._set_line_status[i] = val
        if val == 1:
            self._set_topo_vect[self.line_ex_pos_topo_vect[i]] = 1
            self._set_topo_vect[self.line_or_pos_topo_vect[i]] = 1
        return self
