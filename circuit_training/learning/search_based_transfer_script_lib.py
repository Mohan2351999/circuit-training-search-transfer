# coding=utf-8
# Copyright 2021 The Circuit Training Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sample training with distributed collection using a variable container."""

import os
import time

from typing import Callable
import sys

from absl import logging

from circuit_training.environment import environment as oss_environment
from circuit_training.learning import agent
from circuit_training.learning import learner as learner_lib

import reverb

import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import numpy as np
import random

from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.networks import network
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.train import learner as actor_learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import train_utils
from tf_agents.utils import common
import keras
from tf_agents.train import learner


def train(
    root_dir: str,
    parent_chkpt_dir: str,
    parent_policy_dir: str,
    strategy: tf.distribute.Strategy,
    replay_buffer_server_address: str,
    variable_container_server_address: str,
    create_env_fn: Callable[[], oss_environment.CircuitEnv],
    sequence_length: int,
    actor_net: network.Network,
    value_net: network.Network,
    # Training params
    # This is the per replica batch size. The global batch size can be computed
    # by this number multiplied by the number of replicas (8 in the case of 2x2
    # TPUs).
    use_grl: bool = True,
    per_replica_batch_size: int = 32,
    num_epochs: int = 4,
    num_iterations: int = 10000,
    # This is the number of episodes we train on in each iteration.
    # num_episodes_per_iteration * epsisode_length * num_epochs =
    # global_step (number of gradient updates) * per_replica_batch_size *
    # num_replicas.
    num_episodes_per_iteration: int = 1024,
    allow_variable_length_episodes: bool = False) -> None:
  """Trains a PPO agent.

  Args:
    root_dir: Main directory path where checkpoints, saved_models, and summaries
      will be written to.
    strategy: `tf.distribute.Strategy` to use during training.
    replay_buffer_server_address: Address of the reverb replay server.
    variable_container_server_address: The address of the Reverb server for
      ReverbVariableContainer.
    create_env_fn: Function to create circuit training environment.
    sequence_length: sssFixed sequence length for elements in the dataset. Used for
      calculating how many iterations of minibatches to use for training.
    actor_net: TF-Agents actor network.
    value_net: TF-Agents value network.
    use_grl: Whether to use GRL agent network or RL fully connected agent
      network.
    per_replica_batch_size: The minibatch size for learner. The dataset used for
      training is shaped `[minibatch_size, 1, ...]`. If None, full sequences
      will be fed into the agent. Please set this parameter to None for RNN
      networks which requires full sequences.
    num_epochs: The number of iterations to go through the same sequences. The
      num_episodes_per_iteration are repeated for num_epochs times in a
      particular learner run.
    num_iterations: The number of iterations to run the training.
    num_episodes_per_iteration: This is the number of episodes we train in each
      epoch.
    allow_variable_length_episodes: Whether to support variable length episodes
      for training.
  """
  # Get the specs from the environment.
  print("Started creating the environment!")
  env = create_env_fn()
  _, action_tensor_spec, time_step_tensor_spec = (
      spec_utils.get_tensor_specs(env))

  # Create the agent.
  with strategy.scope():
    train_step = train_utils.create_train_step()
    model_id = common.create_variable('model_id')
    creat_agent_fn = agent.create_circuit_ppo_grl_agent
    tf_agent = creat_agent_fn(
        train_step,
        action_tensor_spec,
        time_step_tensor_spec,
        actor_net,
        value_net,
        strategy,
    )
    tf_agent.initialize()
  
  ########################################################################
  #Pick the most recent checkpoint from the parent directory.
  print("Start loading the latest Checkpoints!")
  parent_checkpoint_dir = parent_chkpt_dir
  lst = os.listdir(parent_checkpoint_dir)
  lst.sort()
  final_checkpoint_dir = os.path.join(parent_checkpoint_dir, lst[-1])

  # print("Loading the latest checkpoint from:", final_checkpoint_dir)

  # Now load the checkpointer
  load_checkpointer = common.Checkpointer(
    ckpt_dir=parent_checkpoint_dir,
    max_to_keep=20,
    agent=tf_agent,
    policy=tf_agent.policy)
  
  ## load the checkpointer
  load_checkpointer.initialize_or_restore()
  # print("Loaded the latest checkpoint from parent model and restored the graph!")

  global_step = tf.compat.v1.train.get_global_step()
  print("The global step is:", global_step)

  obs = env.reset()
  done = False
  count = 1

  wirelength_track = []
  density_track = []
  congestion_track = []
  valid_placement = []
  tf_agent_track = []
  actor_net_track = []
  value_net_track = []
  model_idx_track = []

  print(dir(obs))

  last = False

  # print("--------------------------------", obs, "checking is last:", obs.is_last())
  while not obs.is_last():
      # print("--------------------------------", obs)
      action = tf_agent.policy.action(obs)
      # print("************* ACTION TAKEN", action.action)
      obs= env.step(action.action)
      step_type, reward, discount, observation, = obs
      # print(f"***** Placing node {count}/133 with current reward {reward} and observation {observation} and discount {discount} \n and the info is {env.call_analytical_placer_and_get_cost()}")
      # print(f"Currently at step {count} with obs {obs}\n with info {info}, \n and is done {done} with cost {reward}")
      if obs.is_last():
        last = True
      count+=1
  
  print("******* Base Model info *******")
  print("Congestion values", env.call_analytical_placer_and_get_cost()[1]['congestion'])
  print("Wirelength values", env.call_analytical_placer_and_get_cost()[1]['wirelength'])
  print("Density values", env.call_analytical_placer_and_get_cost()[1]['density'])

  wirelength_track.append(env.call_analytical_placer_and_get_cost()[1]['wirelength'])
  density_track.append(env.call_analytical_placer_and_get_cost()[1]['density'])
  congestion_track.append(env.call_analytical_placer_and_get_cost()[1]['congestion'])
  valid_placement.append(last)
  tf_agent_track.append(tf_agent_track)
  actor_net_track.append(actor_net)
  value_net_track.append(value_net)
  model_idx_track.append('0_0')


  print("Finished placing all the nodes")
 
  # # # 0000000000

  print("Actor Net: ", actor_net._shared_network._model._policy_location_head.summary())
  # print("Value Net:", value_net._shared_network._model._value_head.summary())

  # actornet = actor_net._shared_network._model._policy_location_head
  # valuenet = value_net._shared_network._model._value_head

  # # actornet = actornet.copy()
  # actornet_copy = keras.models.clone_model(actornet)
  # print("***** Succesfully clonned model")
  # # valuenet = valuenet.copy()

  # print("Actor Net: ", actornet_copy.summary())
  # print("Value Net:", valuenet.summary())

  # print("Checking the model weights!")
  # print(actornet_copy.layers[0].get_weights()[0], "the weights shape is:", actornet_copy.layers[0].get_weights()[0].shape)

  # # print("Trying to access ActorNet Model weights")
  # # for i, layer in enumerate(actornet.layers):
  # #   if len(layer.get_weights()) == 2:
  # #     print(f"At layer {i} with layer details: {layer} with weights dimensions {layer.get_weights()[0].shape} and bias dimensions {layer.get_weights()[1].shape}")
  # #   else:
  # #     print(f"At layer {i} with layer details: {layer} with weights dimensions {len(layer.get_weights())}")
  # #   print("------")
  
  # # Sample reference to change the model weights.
  # first_layer_original_weights = np.array(actornet_copy.layers[0].get_weights())

  # first_layer_original_weights[0] = first_layer_original_weights[0] + 1

  # print("The first layer weights(w) is: \n",first_layer_original_weights[0])

  # actornet_copy.layers[0].set_weights(first_layer_original_weights)

  # # actornet.layers[0].get_weights()[0] = actornet.layers[0].get_weights()[0] + 1
  # print("Values after changing the model weights!")
  # print(actornet_copy.layers[0].get_weights()[0], "the weights shape is:", actornet_copy.layers[0].get_weights()[0].shape)

  # # First create the inference for the model with changed weights. 
  # # Step 1: Save the checkpoint

  # sample_save_dir = "logs/run_02/55/child_models/"

  # os.makedirs(sample_save_dir, exist_ok=True)
  
  # sample_model_name_01 = "sample_model_01"
  # sample_model_dir = os.path.join(sample_save_dir, sample_model_name_01)

  # # create the agent with the new actor and value net.

  # actornet_1 = actor_net.copy()
  # valuenet_1 = value_net.copy()

  # # actornet_1._shared_network._model._policy_location_head = actornet_copy
  # # valuenet_1._shared_network._model._value_head = valuenet

  # # print("Created a new cop of the original model", actornet_1)

  # with strategy.scope():
  #   train_step = train_utils.create_train_step()
  #   model_id = common.create_variable('model_id')

  #   logging.info('Using GRL agent networks.')
  #   creat_agent_fn = agent.create_circuit_ppo_grl_agent

  #   print("The actornet copy instance is:", actornet_1, actor_net)

  #   tf_agent_1 = creat_agent_fn(
  #       train_step,
  #       action_tensor_spec,
  #       time_step_tensor_spec,
  #       actornet_1,
  #       valuenet_1,
  #       strategy,
  #   )
  #   tf_agent_1.initialize()
  
  # save_checkpointer = common.Checkpointer(
  #   ckpt_dir=sample_model_dir,
  #   max_to_keep=1,
  #   agent=tf_agent,
  #   policy=tf_agent.policy)

  # # save_checkpointer.save(0)

  # save_checkpointer.initialize_or_restore()
  # print("Loaded the Saved model checkpoint at ", sample_model_dir)

  # # Model weights
  # actornet_test = actornet_1._shared_network._model._policy_location_head
  # valuenet_test = valuenet_1._shared_network._model._value_head

  # print("Model weight after loading from checkpoint", actornet_test.layers[0].get_weights()[0])

  # print("Starting the inference after modifying the weights: ")

  # obs = env.reset()
  # done = False
  # count = 1

  #  # print("--------------------------------", obs, "checking is last:", obs.is_last())
  # while not obs.is_last():
  #     # print("--------------------------------", obs)
  #     action = tf_agent_1.policy.action(obs)
  #     print("************* ACTION TAKEN", action.action)
  #     obs= env.step(action.action)
  #     # print(obs)
  #     discount, observation, reward, step_type = obs
  #     print(f"***** Placing node {count}/133 with current reward {reward} and info {step_type} and discount {discount}")
  #     # print(f"Currently at step {count} with obs {obs}\n with info {info}, \n and is done {done} with cost {reward}")
  #     count+=1
  
  # # sample
  # # 000000000
  
  # Similarly in the search based algorithms - base model created -  now .
  population = 10
  generations = 10

  # wirelength_track
  # density_track
  # congestion_track
  # valid_placement
  # tf_agent_track
  # actor_net_track
  # value_net_track
  # model_idx_trck

  # Multiply mutation operations.
  def mutation_3(policy_network, value_network):
    # Change the weights of a particular conv2d layer.
    network = policy_network._shared_network._model._policy_location_head
    # value_network = value_network._shared_network._model._value_head
    # network = policy_network
    model_config = network.get_config()
    # print("Model config", model_config)
    conv_locations = []

    for i, layer in enumerate(network.layers):
      # print(i, layer, layer.name)
      if(isinstance(layer, keras.layers.Conv2DTranspose)):
        conv_locations.append(i)
        # print("Conv2DTranspose layer weights shape: ", len(layer.get_weights()), layer.get_weights()[0].shape, layer.get_weights()[1].shape)
    
    # print("All the conv layers:", conv_locations)
    # Randomly pick a location.
    random_conv_location = random.choice(conv_locations)
    for i, layer in enumerate(network.layers):
      if i== random_conv_location:
        generated_weights = np.random.uniform(0, 1, size=layer.get_weights()[0].shape)
        new_weights = np.multiply(layer.get_weights()[0], generated_weights)
        assign_weights = [new_weights, layer.get_weights()[1]] # Assigning both weights and bias.
        network.layers[i].set_weights(assign_weights)
        print(f"Multiply weights to a Conv2d layer at index {random_conv_location}")
    
    # Assign the models to policy and value network
    policy_network._shared_network._model._policy_location_head = network
        
    # sys.exit("Exiting program here")
    return policy_network, value_network

  # Subtract mutation operations.
  def mutation_2(policy_network, value_network):
    # Change the weights of a particular conv2d layer.
    network = policy_network._shared_network._model._policy_location_head
    # value_network = value_network._shared_network._model._value_head
    # network = policy_network
    model_config = network.get_config()
    # print("Model config", model_config)
    conv_locations = []

    for i, layer in enumerate(network.layers):
      # print(i, layer, layer.name)
      if(isinstance(layer, keras.layers.Conv2DTranspose)):
        conv_locations.append(i)
        # print("Conv2DTranspose layer weights shape: ", len(layer.get_weights()), layer.get_weights()[0].shape, layer.get_weights()[1].shape)
    
    # print("All the conv layers:", conv_locations)
    # Randomly pick a location.
    random_conv_location = random.choice(conv_locations)
    for i, layer in enumerate(network.layers):
      if i== random_conv_location:
        generated_weights = np.random.uniform(0, 1, size=layer.get_weights()[0].shape)
        new_weights = np.subtract(layer.get_weights()[0], generated_weights)
        assign_weights = [new_weights, layer.get_weights()[1]] # Assigning both weights and bias.
        network.layers[i].set_weights(assign_weights)
        print(f"Subtract weights to a Conv2d layer at index {random_conv_location}")
    
    # Assign the models to policy and value network
    policy_network._shared_network._model._policy_location_head = network
        
    # sys.exit("Exiting program here")
    return policy_network, value_network

  # Add mutation operations.
  def mutation_1(policy_network, value_network):
    # Change the weights of a particular conv2d layer.
    network = policy_network._shared_network._model._policy_location_head
    # value_network = value_network._shared_network._model._value_head
    # network = policy_network
    model_config = network.get_config()
    # print("Model config", model_config)
    conv_locations = []

    for i, layer in enumerate(network.layers):
      # print(i, layer, layer.name)
      if(isinstance(layer, keras.layers.Conv2DTranspose)):
        conv_locations.append(i)
        # print("Conv2DTranspose layer weights shape: ", len(layer.get_weights()), layer.get_weights()[0].shape, layer.get_weights()[1].shape)
    
    # print("All the conv layers:", conv_locations)
    # Randomly pick a location.
    random_conv_location = random.choice(conv_locations)
    for i, layer in enumerate(network.layers):
      if i== random_conv_location:
        generated_weights = np.random.uniform(0, 1, size=layer.get_weights()[0].shape)
        new_weights = np.add(layer.get_weights()[0], generated_weights)
        assign_weights = [new_weights, layer.get_weights()[1]] # Assigning both weights and bias.
        network.layers[i].set_weights(assign_weights)
        print(f"Added weights to a Conv2d layer at index {random_conv_location}")
    
    # Assign the models to policy and value network
    policy_network._shared_network._model._policy_location_head = network
        
    # sys.exit("Exiting program here")
    return policy_network, value_network

  def mutations(actor_model, value_model):
    x = random.randint(0, 2)
    print(f">>>>>>>>>> Choose the mutation {x}")
    if x ==0:
      # Add weights to a layer.
      child_actor_model, child_value_model = mutation_1(actor_model, value_model)
    elif x == 1:
      # Subtract weights to a layer.
      child_actor_model, child_value_model = mutation_2(actor_model, value_model)    
    elif x== 2:
      # Multiply weights to a layer.
      child_actor_model, child_value_model = mutation_3(actor_model, value_model)

    return child_actor_model, child_value_model

  # All the iterations in the  generations.
  for iteration in range(generations):
    print(f"****************************** At the generation {iteration}")
    if iteration==0:
      for i in range(population-1):
        print(f"Currently in generation {iteration} and generating child {i}")
        base_actor_model = actor_net_track[0]
        base_value_model = value_net_track[0]
        child_actor_model, child_value_model = mutations(base_actor_model, base_value_model) 
        actor_net_track.append(child_actor_model)
        value_net_track.append(child_value_model)
        model_idx_track.append(f'{iteration}_{i}')

        env = create_env_fn()
        _, action_tensor_spec, time_step_tensor_spec = (spec_utils.get_tensor_specs(env))
        train_step = train_utils.create_train_step()

        with strategy.scope():
          logging.info('Using GRL agent networks.')
          creat_agent_fn = agent.create_circuit_ppo_grl_agent

          tf_agent_1 = creat_agent_fn(
              train_step,
              action_tensor_spec,
              time_step_tensor_spec,
              child_actor_model,
              child_value_model,
              strategy,
          )
          tf_agent_1.initialize()
        
        tf_agent_track.append(tf_agent_1)

        obs = env.reset()
        count = 1
        last = False

        # print("--------------------------------", obs, "checking is last:", obs.is_last())
        while not obs.is_last():
            # print("--------------------------------", obs)
            action = tf_agent_1.policy.action(obs)
            # print("************* ACTION TAKEN", action.action)
            obs= env.step(action.action)
            # print(obs)
            discount, observation, reward, step_type = obs
            # print(f"***** Placing node {count}/133 with current reward {reward} and info {step_type} and discount {discount}")
            # print(f"Currently at step {count} with obs {obs}\n with info {info}, \n and is done {done} with cost {reward}")
            if obs.is_last():
              last = True
            count+=1
        print(f"Finished placing chip with nodes {count}/133")
        # Now get the inference values density, wirelength and congestion.

        wirelength_track.append(env.call_analytical_placer_and_get_cost()[1]['wirelength'])
        density_track.append(env.call_analytical_placer_and_get_cost()[1]['density'])
        congestion_track.append(env.call_analytical_placer_and_get_cost()[1]['congestion'])
        valid_placement.append(last)
        print("Recorded all the info of the mutated model\n")
      print("Density track:", density_track)
      print("Congestion track:", congestion_track)
      print("Wirelength track:", wirelength_track)
      print("Valid placement:", valid_placement)
    
    else:
      # Pick the best model - Either create a fitness function (or) just write the logic here itself.
      min_cost = 10e6
      best_index = 0
      for i in range(population):
        if valid_placement[i]:
          if min_cost > (density_track[i] + congestion_track[i] + wirelength_track[i]):
            min_cost = density_track[i] + congestion_track[i] + wirelength_track[i]
            best_index = i
      
      # Delete all the other information of other candidates.
      wirelength_track = wirelength_track[:best_index+1]
      wirelength_track = wirelength_track[best_index:]

      density_track = density_track[:best_index+1]
      density_track = density_track[best_index:]

      congestion_track = congestion_track[:best_index+1]
      congestion_track = congestion_track[best_index:]

      valid_placement = valid_placement[:best_index+1]
      valid_placement = valid_placement[best_index:]

      tf_agent_track = tf_agent_track[:best_index+1]
      tf_agent_track = tf_agent_track[best_index:]

      actor_net_track = actor_net_track[:best_index+1]
      actor_net_track = actor_net_track[best_index:]

      value_net_track = value_net_track[:best_index+1]
      value_net_track = value_net_track[best_index:]

      model_idx_track = model_idx_track[:best_index+1]
      model_idx_track = model_idx_track[best_index:]

      print(f"$$$$$$$$$$$$$$$ Found the best index at {best_index}")
      print(f"Deleting the remiaing models and info, current size is:", len(tf_agent_track))

      print(f"******* Base Model info  in generation {iteration} *******")
      print("Congestion values", congestion_track[0])
      print("Wirelength values", wirelength_track[0])
      print("Density values", density_track[0])

      # generate the models similar to the first generation.
      for i in range(population-1):
        print(f"Currently in generation {iteration} and generating child {i}")
        base_actor_model = actor_net_track[0]
        base_value_model = value_net_track[0]
        child_actor_model, child_value_model = mutations(base_actor_model, base_value_model) 
        actor_net_track.append(child_actor_model)
        value_net_track.append(child_value_model)
        model_idx_track.append(f'{iteration}_{i}')

        env = create_env_fn()
        _, action_tensor_spec, time_step_tensor_spec = (spec_utils.get_tensor_specs(env))
        train_step = train_utils.create_train_step()

        with strategy.scope():
          logging.info('Using GRL agent networks.')
          creat_agent_fn = agent.create_circuit_ppo_grl_agent

          tf_agent_1 = creat_agent_fn(
              train_step,
              action_tensor_spec,
              time_step_tensor_spec,
              child_actor_model,
              child_value_model,
              strategy,
          )
          tf_agent_1.initialize()
        
        tf_agent_track.append(tf_agent_1)

        obs = env.reset()
        count = 1
        last = False

        # print("--------------------------------", obs, "checking is last:", obs.is_last())
        while not obs.is_last():
            # print("--------------------------------", obs)
            action = tf_agent_1.policy.action(obs)
            # print("************* ACTION TAKEN", action.action)
            obs= env.step(action.action)
            # print(obs)
            discount, observation, reward, step_type = obs
            # print(f"***** Placing node {count}/133 with current reward {reward} and info {step_type} and discount {discount}")
            # print(f"Currently at step {count} with obs {obs}\n with info {info}, \n and is done {done} with cost {reward}")
            if obs.is_last():
              last = True
            count+=1
        print(f"Finished placing chip with nodes {count}/133")
        # Now get the inference values density, wirelength and congestion.

        wirelength_track.append(env.call_analytical_placer_and_get_cost()[1]['wirelength'])
        density_track.append(env.call_analytical_placer_and_get_cost()[1]['density'])
        congestion_track.append(env.call_analytical_placer_and_get_cost()[1]['congestion'])
        valid_placement.append(last)

      print("Density track:", density_track)
      print("Congestion track:", congestion_track)
      print("Wirelength track:", wirelength_track)