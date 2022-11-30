<!--* freshness: {
  owner: 'agoldie'
  owner: 'azalia'
  owner: 'tobyboyd'
  owner: 'sguada'
  owner: 'morpheus-oss-team'
  reviewed: '2022-01-11'
  review_interval: '12 months'
} *-->

# Circuit Training: An open-source framework for generating chip floor plans with distributed deep reinforcement learning.

*Circuit Training* is an open-source framework for generating chip floor plans
with distributed deep reinforcement learning. This framework reproduces the
methodology published in the Nature 2021 paper:

*[A graph placement methodology for fast chip design.](https://www.nature.com/articles/s41586-021-03544-w)
Azalia Mirhoseini, Anna Goldie, Mustafa Yazgan, Joe Wenjie Jiang, Ebrahim
Songhori, Shen Wang, Young-Joon Lee, Eric Johnson, Omkar Pathak, Azade Nazi,
Jiwoo Pak, Andy Tong, Kavya Srinivasa, William Hang, Emre Tuncer, Quoc V. Le,
James Laudon, Richard Ho, Roger Carpenter & Jeff Dean, 2021. Nature, 594(7862),
pp.207-212.
[[PDF]](https://www.nature.com/articles/s41586-021-03544-w.epdf?sharing_token=tYaxh2mR5EozfsSL0WHZLdRgN0jAjWel9jnR3ZoTv0PW0K0NmVrRsFPaMa9Y5We9O4Hqf_liatg-lvhiVcYpHL_YQpqkurA31sxqtmA-E1yNUWVMMVSBxWSp7ZFFIWawYQYnEXoBE4esRDSWqubhDFWUPyI5wK_5B_YIO-D_kS8%3D)*

Circuit training is built on top of
[TF-Agents](https://github.com/tensorflow/agents) and
[TensorFlow 2.x](https://www.tensorflow.org/) with support for eager execution,
distributed training across multiple GPUs, and distributed data collection
scaling to 100s of actors.

## Table of contents

<a href='#Features'>Features</a><br>
<a href='#Installation'>Installation</a><br>
<a href='#QuickStart'>Quick start</a><br>
<a href='#Results'>Results</a><br>

<a id='Installation'></a>

## Installation

Circuit Training requires:

*   Installing TF-Agents which includes Reverb and TensorFlow.
*   Downloading the placement cost binary into your system path.
*   Downloading the circuit-training code.

Using the code at `HEAD` with the nightly release of TF-Agents is recommended.

```shell
# Installs TF-Agents with nightly versions of Reverb and TensorFlow 2.x
$  pip install tf-agents-nightly[reverb]
# Copies the placement cost binary to /usr/local/bin and makes it executable.
$  sudo curl https://storage.googleapis.com/rl-infra-public/circuit-training/placement_cost/plc_wrapper_main \
     -o  /usr/local/bin/plc_wrapper_main
$  sudo chmod 555 /usr/local/bin/plc_wrapper_main
# Clones the circuit-training repo.
$  git clone https://github.com/google-research/circuit-training.git
```

For Creating Conda environment refer to the for installing using [YML](./conda_environment.yml) file 

<a id='QuickStart'></a>

## Quick start

This quick start places the Ariane RISC-V CPU macros by training the deep
reinforcement policy from scratch. The `num_episodes_per_iteration` and
`global_batch_size` used below were picked to work on a single machine training
on CPU. The purpose is to illustrate a running system, not optimize the result.
The result of a few thousand steps is shown in this
[tensorboard](https://tensorboard.dev/experiment/r1Xn1pD3SGKTGyo64saeaw). The
full scale Ariane RISC-V experiment matching the paper is detailed in
[Circuit training for Ariane RISC-V](./docs/ARIANE.md).

There are many open source like LEF/DEF, Bookshelf formats. But, this repository 
expects you to convert everything into Protobuf Format. Refer to [MacroPlacement](https://github.com/TILOS-AI-Institute/MacroPlacement/tree/main/CodeElements/FormatTranslators) repo
for the converters(to convert to protobuf format).

# Training Model

The following jobs will be created by the steps below:

*   1 Replay Buffer (Reverb) job
*   1-3 Collect jobs
*   1 Train job
*   1 Eval job

Each job is started in a `tmux` session. To switch between sessions use `ctrl +
b` followed by `s` and then select the specified session.

```shell
# Sets the environment variables needed by each job. These variables are
# inherited by the tmux sessions created in the next step.
$  export ROOT_DIR=./logs/run_00
$  export REVERB_PORT=8008
$  export REVERB_SERVER="127.0.0.1:${REVERB_PORT}"
$  export NETLIST_FILE=./circuit_training/environment/test_data/ariane/netlist.pb.txt
$  export INIT_PLACEMENT=./circuit_training/environment/test_data/ariane/initial.plc

# Creates all the tmux sessions that will be used.
$  tmux new-session -d -s reverb_server && \
   tmux new-session -d -s collect_job_00 && \
   tmux new-session -d -s collect_job_01 && \
   tmux new-session -d -s collect_job_02 && \
   tmux new-session -d -s train_job && \
   tmux new-session -d -s eval_job && \
   tmux new-session -d -s tb_job

# Starts the Replay Buffer (Reverb) Job
$  tmux attach -t reverb_server
$  python3 -m circuit_training.learning.ppo_reverb_server \
   --root_dir=${ROOT_DIR}  --port=${REVERB_PORT}

# Starts the Training job
# Change to the tmux session `train_job`.
# `ctrl + b` followed by `s`
$  python3 -m circuit_training.learning.train_ppo \
  --root_dir=${ROOT_DIR} \
  --replay_buffer_server_address=${REVERB_SERVER} \
  --variable_container_server_address=${REVERB_SERVER} \
  --num_episodes_per_iteration=16 \
  --global_batch_size=64 \
  --netlist_file=${NETLIST_FILE} \
  --init_placement=${INIT_PLACEMENT}

# Starts the Collect job
# Change to the tmux session `collect_job_00`.
# `ctrl + b` followed by `s`
$  python3 -m circuit_training.learning.ppo_collect \
  --root_dir=${ROOT_DIR} \
  --replay_buffer_server_address=${REVERB_SERVER} \
  --variable_container_server_address=${REVERB_SERVER} \
  --task_id=0 \
  --netlist_file=${NETLIST_FILE} \
  --init_placement=${INIT_PLACEMENT}

# Starts the Eval job
# Change to the tmux session `eval_job`.
# `ctrl + b` followed by `s`
$  python3 -m circuit_training.learning.eval \
  --root_dir=${ROOT_DIR} \
  --variable_container_server_address=${REVERB_SERVER} \
  --netlist_file=${NETLIST_FILE} \
  --init_placement=${INIT_PLACEMENT}

# Start Tensorboard.
# Change to the tmux session `tb_job`.
# `ctrl + b` followed by `s`
$  tensorboard dev upload --logdir ./logs

# <Optional>: Starts 2 more collect jobs to speed up training.
# Change to the tmux session `collect_job_01`.
# `ctrl + b` followed by `s`
$  python3 -m circuit_training.learning.ppo_collect \
  --root_dir=${ROOT_DIR} \
  --replay_buffer_server_address=${REVERB_SERVER} \
  --variable_container_server_address=${REVERB_SERVER} \
  --task_id=1 \
  --netlist_file=${NETLIST_FILE} \
  --init_placement=${INIT_PLACEMENT}

# Change to the tmux session `collect_job_02`.
# `ctrl + b` followed by `s`
$  python3 -m circuit_training.learning.ppo_collect \
  --root_dir=${ROOT_DIR} \
  --replay_buffer_server_address=${REVERB_SERVER} \
  --variable_container_server_address=${REVERB_SERVER} \
  --task_id=2 \
  --netlist_file=${NETLIST_FILE} \
  --init_placement=${INIT_PLACEMENT}

```

<a id='Zero-shot Learning'></a>

## Zero-shot execution.

For the source task we use the google's ariane circuit and for the target task we use the [Ariane133](https://github.com/TILOS-AI-Institute/MacroPlacement/tree/main/Testcases/ariane133) circuit. For the inference on Ariane133

```shell
python3 -m circuit_training.load_and_restore_model
```

```shell
$ export ROOT_DIR=./logs/run_02 && \
  export REVERB_PORT=8008 && \
  export REVERB_SERVER="127.0.0.1:${REVERB_PORT}" && \
  export NETLIST_FILE=./circuit_training/environment/test_data/ariane133/netlist.pb.txt && \
  export INIT_PLACEMENT=./circuit_training/environment/test_data/ariane133/initial.plc && \
  export CHKPT_DIR=./logs/run_00/111/train/checkpoints/&& \
  export PLCY_DIR=./logs/run_01/111/policies

```

<a id='Finetuning Execution'></a>

## Fine-tuning execution

```shell
$ export ROOT_DIR=./logs/run_03 && \
  export REVERB_PORT=8008 && \
  export REVERB_SERVER="127.0.0.1:${REVERB_PORT}" && \
  export NETLIST_FILE=./circuit_training/environment/test_data/ariane133/netlist.pb.txt && \
  export INIT_PLACEMENT=./circuit_training/environment/test_data/ariane133/initial.plc && \
  export CHKPT_DIR=./logs/run_00/111/train/checkpoints/&& \
  export PLCY_DIR=./logs/run_00/111/policies
  
  # For finetuning on Ariane133 by transferring the weights from Google's ariane.
  python3 -m circuit_training.learning.fine_tuning_script\
  --root_dir=${ROOT_DIR} \
  --parent_chkpt_dir=${CHKPT_DIR} \
  --parent_policy_dir=${PLCY_DIR} \
  --replay_buffer_server_address=${REVERB_SERVER} \
  --variable_container_server_address=${REVERB_SERVER} \
  --num_episodes_per_iteration=16 \
  --global_batch_size=64 \
  --netlist_file=${NETLIST_FILE} \
  --init_placement=${INIT_PLACEMENT}

```

<a id='Search based transfer learning'></a>

# Search based transfer lerning

```shell
$ export ROOT_DIR=./logs/run_03 && \
  export REVERB_PORT=8008 && \
  export REVERB_SERVER="127.0.0.1:${REVERB_PORT}" && \
  export NETLIST_FILE=./circuit_training/environment/test_data/ariane133/netlist.pb.txt && \
  export INIT_PLACEMENT=./circuit_training/environment/test_data/ariane133/initial.plc && \
  export CHKPT_DIR=./logs/run_00/111/train/checkpoints/&& \
  export PLCY_DIR=./logs/run_00/111/policies
  
  # Source model trained on Google Ariane and search based transfer learning on Ariane133
  python3 -m circuit_training.learning.search_based_transfer_script\
  --root_dir=${ROOT_DIR} \
  --parent_chkpt_dir=${CHKPT_DIR} \
  --parent_policy_dir=${PLCY_DIR} \
  --replay_buffer_server_address=${REVERB_SERVER} \
  --variable_container_server_address=${REVERB_SERVER} \
  --num_episodes_per_iteration=16 \
  --global_batch_size=64 \
  --netlist_file=${NETLIST_FILE} \
  --init_placement=${INIT_PLACEMENT}

```
