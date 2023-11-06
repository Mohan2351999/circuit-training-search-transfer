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
from tf_agents.train import learner as actor_learner
from tf_agents.train import triggers
from tf_agents.experimental.distributed import reverb_variable_container


# testcase path
_NETLIST_FILE = flags.DEFINE_string('netlist_file', "./circuit_training/environment/test_data/ariane133/netlist.pb.txt",
                                    'File path to the netlist file.')
_INIT_PLACEMENT = flags.DEFINE_string('init_placement', "./circuit_training/environment/test_data/ariane133/initial.plc",
                                      'File path to the init placement file.')
_OUTPUT_PATH = flags.DEFINE_string('output_path', "./test_placement.plc", "File path to the output placement file.")
_GLOBAL_SEED = flags.DEFINE_integer(
    'global_seed', 55,
    'Used in env and weight initialization, does not impact action sampling.')

greedy_policy_dir = "/home/singamse/Mohan/circuit-training/logs/run_00/55/policies/policy/"

FLAGS = flags.FLAGS

_ROOT_DIR = flags.DEFINE_string('root_dir', "./root_dir/run_finetune", "Path to the saved policy directory")

#Load the pretrained model weights,

# Load the
def main(_argv):
    logging.info('global seed=%d', _GLOBAL_SEED.value)
    logging.info('netlist_file=%s', _NETLIST_FILE.value)
    logging.info('init_placement=%s', _INIT_PLACEMENT.value)
    logging.info('saved_policy_dir=%s', _ROOT_DIR.value)

    root_dir = os.path.join(_ROOT_DIR.value, str(_GLOBAL_SEED.value))

    # initialize test environment
    create_env_fn = functools.partial(
        environment.create_circuit_environment,
        netlist_file=_NETLIST_FILE.value,
        init_placement=_INIT_PLACEMENT.value)
    
    env = create_env_fn()

    observation_tensor_spec, action_tensor_spec, time_step_tensor_spec = (
      spec_utils.get_tensor_specs(env))

    ## Load the architecture of the policy and value net.
    grl_actor_net, grl_value_net = model.create_grl_models(
        observation_tensor_spec,
        action_tensor_spec,
        static_features,
        strategy,
        use_model_tpu=False)
    
    # Create the agent.
    with strategy.scope():
        train_step = train_utils.create_train_step()
        model_id = common.create_variable('model_id')

        creat_agent_fn = agent.create_circuit_ppo_grl_agent

        tf_agent = creat_agent_fn(
            train_step,
            action_tensor_spec,
            time_step_tensor_spec,
            grl_actor_net,
            grl_value_net,
            strategy,
        )
        tf_agent.initialize()
    
    # Create the policy saver which saves the initial model now, then it
    # periodically checkpoints the policy weights.
    saved_model_dir = os.path.join(root_dir, actor_learner.POLICY_SAVED_MODEL_DIR)
    save_model_trigger = triggers.PolicySavedModelTrigger(
        saved_model_dir,
        tf_agent,
        train_step,
        start=-num_episodes_per_iteration,
        interval=num_episodes_per_iteration)
    
    # # Load the pretrained greedy policy from the saved model  
    # saved_greedy_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
    #     greedy_policy_dir, load_specs_from_pbtxt=True)

    # Create the variable container.
    variables = {
        reverb_variable_container.POLICY_KEY: tf_agent.collect_policy.variables(),
        reverb_variable_container.TRAIN_STEP_KEY: train_step,
        'model_id': model_id,
    }
    variable_container   = reverb_variable_container.ReverbVariableContainer(
        variable_container_server_address,
        table_names=[reverb_variable_container.DEFAULT_TABLE])
    variable_container.push(variables)

    # Create the replay buffer.
    reverb_replay_train = reverb_replay_buffer.ReverbReplayBuffer(
        tf_agent.collect_data_spec,
        sequence_length=None,
        table_name='training_table',
        server_address=replay_buffer_server_address)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass