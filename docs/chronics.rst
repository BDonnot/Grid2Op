.. currentmodule:: grid2op.Chronics

.. _time-series-module:

Time series (formerly called "chronics")
=========================================

This page is organized as follow:

.. contents:: Table of Contents
    :depth: 3

Objectives
-----------
This module is present to handle everything related to input data that are not structural.

In the Grid2Op vocabulary a "GridValue" or "Chronics" is something that provides data to change the input parameter
of a power flow between 1 time step and the other.

It is a more generic terminology. Modification that can be performed by :class:`GridValue` object includes, but
are not limited to:

  - injections such as:

    - generators active production setpoint
    - generators voltage setpoint
    - loads active consumption
    - loads reactive consumption

  - structural informations such as:

    - planned outage: powerline disconnection anticipated in advance
    - hazards: powerline disconnection that cannot be anticipated, for example due to a windstorm.

All powergrid modification that can be performed using an :class:`grid2op.Action.BaseAction` can be implemented as
form of a :class:`GridValue`.

The same mechanism than for :class:`grid2op.Action.BaseAction` or :class:`grid2op.Observation.BaseObservation`
is pursued here. All state modifications made by the :class:`grid2op.Environment` must derived from
the :class:`GridValue`. It is not recommended to create them directly, but rather to use
the :class:`ChronicsHandler` for such a purpose.

Note that the values returned by a :class:`GridValue` are **backend dependant**. A GridValue object should always
return the data in the order expected by the :class:`grid2op.Backend`, regardless of the order in which data are given
in the files or generated by the data generator process.

This implies that changing the backend will change the output of :class:`GridValue`. More information about this
is given in the description of the :func:`GridValue.initialize` method.

Finally, compared to other Reinforcement Learning problems, is the possibility to use "forecast". This optional feature
can be accessed via the :class:`grid2op.Observation.BaseObservation` and mainly the
:func:`grid2op.Observation.BaseObservation.simulate` method. The data that are used to generate this forecasts
come from the :class:`grid2op.GridValue` and are detailed in the
:func:`GridValue.forecasts` method.


More control on the time series
-------------------------------
We explained, in the description of the :class:`grid2op.Environment` in sections
:ref:`environment-module-chronics-info` and following how to have more control on which chronics is used,
with steps are used within a chronics etc. We will not detailed here again, please refer to this page
for more information.

However, know that you can have a very detailed control on which time series using the `options`
kwargs of a call to `env.reset()` (or the `reset_otions` kwargs when calling the 
`runner.run()`) : 


Use a specific time serie for an episode
*******************************************

To use a specific time series for a given episode, you can use 
`env.reset(options={"time serie id": THE_ID_YOU_WANT)`.

For example:

.. code-block:: python

  import grid2op
  env_name = "l2rpn_case14_sandbox"
  env = grid2op.make(env_name)

  # you can use an int:
  obs = env.reset(options={"time serie id": 0})

  # or the name of the folder (for most grid2op environment)
  obs = env.reset(options={"time serie id": "0000"})  # for l2rpn_case14_sandbox

  # for say l2rpn_neurips_2020_track1
  # obs = env.reset(options={"time serie id": "Scenario_august_008"})

  # for say l2rpn_idf_2023
  # obs = env.reset(options={"time serie id": "2035-04-23_7"})


.. note::
  For oldest grid2op versions (please upgrade if that's the case) you needed to use:
  `env.set_id(THE_CHRONIC_ID)` (see :func:`grid2op.Environment.Environment.set_id`) to set the id of the
  chronics you want to use.


Skipping the initial few steps
*******************************

Often the time series provided for an environment always start at the same date and time on 
the same hour of the day and day of the week. It might not be ideal to learn controler
with such data or might "burn up" computation time during evaluation.

To do that, you can use the `"init ts"` reset options, for example with:

.. code-block:: python

  import grid2op
  env_name = "l2rpn_case14_sandbox"
  env = grid2op.make(env_name)

  # you can use an int:
  obs = env.reset(options={"init ts": 12})

  # obs will skip the first hour of the time series
  # 12 steps is equivalent to 1h (5 mins per step in general)


.. note::
  
  For oldest grid2op versions (please upgrade if that's the case) you needed to use:
  `env.fast_forward_chronics(nb_time_steps)`
  (see :func:`grid2op.Environment.BaseEnv.fast_forward_chronics`) to skip initial 
  few steps
  of a given chronics.

  Please be aware that this "legacy" behaviour has some issues and is "less clear"
  than the "init ts" above and it can have some weird combination with 
  `set_max_iter` for example.


Limit the maximum length of the current episode
*************************************************

For most enviroment, the maximum duration of an episode is the equivalent of a week
(~2020 steps) or a month (~8100 steps) which might be too long for some usecase.

Anyway, if you want to reduce it, you can now do it with the `"max step"` reset
option like this:

.. code-block:: python

  import grid2op
  env_name = "l2rpn_case14_sandbox"
  env = grid2op.make(env_name)

  # you can use an int:
  obs = env.reset(options={"max step": 2*288})

  # the maximum duration of the episode is now 2*288 steps
  # the equivalent of two days

.. note::
  
  For oldest grid2op versions (please upgrade if that's the case) you needed to use:
  `env.chronics_handler.set_max_iter(nb_max_iter)`
  (see :func:`grid2op.Chronics.ChronicsHandler.set_max_iter`) to limit the number 
  of steps within an episode.

  Please be aware that this "legacy" behaviour has some issues and is "less clear"
  than the "init ts" above and it can have some weird combination with 
  `fast_forward_chronics` for example.

Discard some time series from the existing folder
**************************************************

The folder containing the time series for a given grid2op environment often contains
dozens (thousands sometimes) different time series.

You might want to use only part of them at some point (whether it's some for training and some
for validation and test, or some for training an agent on a process and some to train the 
same agent on another process etc.)

Anyway, if you want to do this (on the majority of released environments) you can do it
thanks to the `env.chronics_handler.set_filter(a_function)`.

For example:

.. code-block:: python

  import re
  import grid2op
  env_name = "l2rpn_case14_sandbox"
  env = grid2op.make(env_name)

  def keep_only_some_ep(chron_name):
    return re.match(r".*00.*", chron_name) is not None

  env.chronics_handler.set_filter(keep_only_some_ep)
  li_episode_kept = env.chronics_handler.reset()


.. note::
  For oldest grid2op versions (please upgrade if that's the case) you needed to use:
  use `env.chronics_handler.set_filter(a_function)` (see :func:`grid2op.Chronics.GridValue.set_filter`)
  to only use certain chronics


- use `env.chronics_handler.sample_next_chronics(probas)`
  (see :func:`grid2op.Chronics.GridValue.sample_next_chronics`) to draw at random some chronics

Performance gain (throughput)
********************************

Chosing the right chronics can also lead to some large advantage in terms of computation time. This is
particularly true if you want to benefit the most from HPC for example. More detailed is given in the
:ref:`environment-module-data-pipeline` section. In summary:

- set the "chunk" size (amount of data read from the disk, instead of reading an entire scenarios, you read
  from the hard drive only a certain amount of data at a time, see
  :func:`grid2op.Chronics.ChronicsHandler.set_chunk_size`) you can use it with
  `env.chronics_handler.set_chunk_size(100)`
- cache all the chronics and use them from memory (instead of reading them from the hard drive, see
  :class:`grid2op.Chronics.MultifolderWithCache`) you can do this with
  `env = grid2op.make(..., chronics_class=MultifolderWithCache)`

Finally, if you need to study machine learning in a "regular" fashion, with a train / validation / set
you can use the `env.train_val_split` or `env.train_val_split_random` functions to do that. See
an example usage in the section :ref:`environment-module-train-val-test`.




Detailed Documentation by class
--------------------------------

.. automodule:: grid2op.Chronics
    :members:
    :autosummary:

.. include:: final.rst
