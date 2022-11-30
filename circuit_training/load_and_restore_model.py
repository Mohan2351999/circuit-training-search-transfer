import tensorflow as tf
import functools
import os
from absl import app
from absl import flags
from absl import logging
import numpy as np
import reverb
print('current directory', os.getcwd())

from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import train_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.utils import common
from tf_agents.trajectories import TimeStep
from tf_agents.environments import suite_gym
from circuit_training.environment import environment
from circuit_training.environment import plc_client
from circuit_training.model import model
from circuit_training.learning import agent


""" Execute the command python3 -m circuit_training.load_and_restore_model
    to run the notebook.
"""

# testcase path
_NETLIST_FILE = flags.DEFINE_string('netlist_file', "./circuit_training/environment/test_data/ariane133/netlist.pb.txt",
                                    'File path to the netlist file.')
_INIT_PLACEMENT = flags.DEFINE_string('init_placement', "./circuit_training/environment/test_data/ariane133/initial.plc",
                                      'File path to the init placement file.')
_OUTPUT_PATH = flags.DEFINE_string('output_path', "./test_placement.plc", "File path to the   placement file.")
_GLOBAL_SEED = flags.DEFINE_integer(
    'global_seed', 111,
    'Used in env and weight initialization, does not impact action sampling.')

greedy_policy_dir = "/home/singamse/Mohan/circuit-training/logs/run_01/111/policies/policy/"

FLAGS = flags.FLAGS

def main(_argv):
    # ##### Random Testing ##
    # test_netlist_dir = ('circuit_training/'
    #                     'environment/test_data/ariane')
    # netlist_file = os.path.join(FLAGS.test_srcdir, test_netlist_dir,
    #                             'netlist.pb.txt')
    # init_placement = os.path.join(FLAGS.test_srcdir, test_netlist_dir,
    #                               'initial.plc')
    # env = environment.CircuitEnv(
    #     netlist_file=netlist_file,
    #     init_placement=init_placement,)
    # obs = env.reset()
    
    # # Load the policy 
    # saved_greedy_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
    #     greedy_policy_dir, load_specs_from_pbtxt=True)

    # # saved_greedy_policy = tf.saved_model.load(greedy_policy_dir)

    # print("Loaded the model!")

    # ## Load the architecture of the policy and value net.
    # # grl_actor_net, grl_value_net = model.create_grl_models(
    # #     observation_tensor_spec,
    # #     action_tensor_spec,
    # #     static_features,
    # #     strategy,
    # #     use_model_tpu=False)

    # obs = env.reset()
    # done = False
    # count = 0
    # print("--------------------------------", obs)
    # # print(obs['discount'])
    # while not done:
    #     print("--------------------------------", obs)
    #     action = saved_greedy_policy.action(obs['mask'])
    #     print("************* ACTION TAKEN", action)
    #     obs, reward, done, info = env.step(action)
    #     print(f"Currently at step {count}: with info {info1}")
    #     count+=1
    ##############################

    logging.info('global seed=%d', _GLOBAL_SEED.value)
    logging.info('netlist_file=%s', _NETLIST_FILE.value)
    logging.info('init_placement=%s', _INIT_PLACEMENT.value)

    # initialize test environment
    create_env_fn = functools.partial(
        environment.create_circuit_environment,
        netlist_file=_NETLIST_FILE.value,
        init_placement=_INIT_PLACEMENT.value)

    test_env = create_env_fn()

    # print("The current state space is:")
    # print(test_env.time_step_spec())

    # print("The current action space is:")
    # print(test_env.action_spec())

    observation_tensor_spec, action_tensor_spec, _ = (
      spec_utils.get_tensor_specs(test_env))
    
    static_features = test_env.wrapped_env().get_static_obs()

    # print("Static feataures inputed to the model: ", static_features)

    strategy = strategy_utils.get_strategy(FLAGS.tpu, FLAGS.use_gpu)

    # print("Strategy used: ", strategy)

    print("Before loading model!!")
    # Load the policy 
    saved_greedy_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
        greedy_policy_dir, load_specs_from_pbtxt=True)

    # saved_greedy_policy = tf.saved_model.load(greedy_policy_dir)

    print("Loaded the model!")

    ## Load the architecture of the policy and value net.
    grl_actor_net, grl_value_net = model.create_grl_models(
        observation_tensor_spec,
        action_tensor_spec,
        static_features,
        strategy,
        use_model_tpu=False)

    ## Find the wirelength and congestion and denesity values at the inference.
    obs = test_env.reset()
    done = False
    count = 1
    print("--------------------------------", obs, "checking is last:", obs.is_last())
    while not obs.is_last():
        # print("--------------------------------", obs)
        action = saved_greedy_policy.action(obs)
        print("************* ACTION TAKEN", action.action)
        obs = test_env.step(action.action)
        discount, observation, reward, step_type = obs
        print(f"***** Placing node {count}/133 with current reward {reward} and info {step_type} and discount {discount}")
        # print(f"Currently at step {count} with obs {obs}\n with info {info}, \n and is done {done} with cost {reward}")
        count+=1
    
    print("Finished placing nodes!")
    
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass