import grid2op
import numpy as np
import pdb
import sys
env = grid2op.make('wcci_test')
np.set_printoptions(precision=2)
# Run Game from specific ts
# Compare redispatching VS DoNothing
t0, tf = 7163, 7168

# Run Do nothing
# print ('\nDo nothing results ===')
# env.chronics_handler.tell_id(183)
# obs = env.reset()
# print(env.chronics_handler.get_name())
#
# env.fast_forward_chronics(t0)
# for i in range(tf-t0+1):
#     action = {} # Do nothing
#     obs, reward, done, info = env.step(env.action_space(action))
#     print ('Hydro MW: {:2f} MW at ts {:1d}'.format(obs.prod_p[8], t0+i))

# Run Redispatching
env.chronics_handler.tell_id(183)
obs = env.reset()
print("Env preparation (start enough generators)")
# stop before the time step to start up some generators
id_before = 20
env.fast_forward_chronics(t0-id_before)
obs = env.get_obs()

# start each generator that can be started
gen_disp_orig = [2, 3, 10, 13, 16]  # some dispatchable generators
ratio = 0.3  # fine tune to max out the production available
array_before = [(el, ratio * env.gen_max_ramp_up[el]) for el in gen_disp_orig]
act2 = env.action_space({'redispatch': array_before})
for i in range(id_before):
    # print(act2)
    prev_p = obs.prod_p
    obs, reward, done, info = env.step(act2)
    this_p = obs.prod_p
    # if info["exception"]:
    #     print("ERROR for ts {}".format(i))
    #     print(info["exception"])
    #     pdb.set_trace()
    #     sys.exit(1)
print("I turned on +{:.2f}MW before starting to ramp up the dump".format(np.sum(obs.prod_p[gen_disp_orig])))
print("Generators productions are {}".format(obs.prod_p))

print('\nRedispatching results === ->action +10MW every time step')
dispatch_val = 8
new_ratio = - ratio * dispatch_val / np.sum(act2._redispatch)
array_before = [(el, new_ratio * env.gen_max_ramp_up[el]) for el in gen_disp_orig]
act2 = env.action_space({'redispatch': array_before})

print("actual dispatch init: {}".format(obs.actual_dispatch[gen_disp_orig]))
for i in range(tf-t0+1):
    print("__________________")
    # do the hydro action
    action = env.action_space()
    # action = env.action_space({'redispatch': [(8, dispatch_val)]})
    # compensate by reducing the start up generators
    # if np.sum(obs.prod_p[gen_disp_orig]) > 10.:
    #     # array_before = [(el, ratio * env.gen_max_ramp_up[el]) for el in gen_disp_orig]
    #     action += act2
    prev_p = obs.prod_p
    obs, reward, done, info = env.step(action)
    this_p = obs.prod_p
    print("\tactual dispatch at ts {}: {}".format(i, obs.actual_dispatch[gen_disp_orig]))
    print('\tHydro MW: {:2f} MW at ts {:1d}'.format(obs.prod_p[8], t0+i))
    print('\tenv target dispatch for hydro {}'.format(obs.target_dispatch[8]))
    if info["exception"]:
        print (info['exception'])
    #     pdb.set_trace()
    #     break
