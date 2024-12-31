# Introduction

[Jittor](https://cg.cs.tsinghua.edu.cn/jittor/) implementation of [TCP](https://github.com/OpenDriveLab/TCP) and [AD-MLP](https://github.com/E2E-AD/AD-MLP) in [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) benchmark.

# Checkpoint
- TCP trained on Bench2Drive base set
    - [Hugging Face Link](https://huggingface.co/rethinklab/Bench2DriveZoo/tree/main)
    - [Baidu Cloud Link](https://pan.baidu.com/s/1CgYscY2esIJLRepkO3FBvQ?pwd=1234)
- ADMLP trained on Bench2Drive base set
    - [Hugging Face Link](https://huggingface.co/rethinklab/Bench2DriveZoo/tree/main)
    - [Baidu Cloud Link](https://pan.baidu.com/s/1RefJxk0B4kYcnf63Vi-ISA?pwd=1234)

Note that for TCP and AD-MLP, you should download the checkpoints trained by Jittor - tcp_b2d_jittor.pkl and admlp_b2d_jittor.pkl.

# Get Started

## Create Environment:
```
## python3.8 should be strictly followed.
conda create -n b2d_zoo_tcp python=3.8
conda activate b2d_zoo_tcp
sudo apt install libomp-dev
pip install git+https://github.com/Jittor/jittor.git
## If you have a GPU and want to enable CUDA acceleration, install CUDA to the Jittor cache
python -m jittor_utils.install_cuda
pip install imgaug 
```

## Prepare Data (If training is required)
1. Download dataset from [Bench2Drive official repo](https://github.com/Thinklab-SJTU/Bench2Drive)

2. Generate .npy data

```bash
# TCP
# Need set your Bench2Drive data path (base or full) in the follwoing file
python tools/gen_tcp_data.py
```
```bash
# ADMLP
# Need set your Bench2Drive data path (base or full) in the follwoing file
python tools/gen_admlp_data.py
```

# Train
First, set the dataset path in TCP/config.py or ADMLP/config.py. Training:
```bash
# TCP
export PYTHONPATH=$PYTHONPATH:PATH_TO_TCP
mpirun -np 1 python TCP/train.py
# or
bash TCP/train.sh # need set your PATH_TO_TCP
# ADMLP
export PYTHONPATH=$PYTHONPATH:PATH_TO_ADMLP
mpirun -np 1 python ADMLP/train.py
# or
bash ADMLP/train.sh # need set your PATH_TO_ADMLP
```

# Open Loop Evaluation
```bash
# TCP, need to specify the checkpoint path in test.py
export PYTHONPATH=$PYTHONPATH:PATH_TO_TCP
mpirun -np 1 python TCP/test.py
# ADMLP, need to specify the checkpoint path in test.py
export PYTHONPATH=$PYTHONPATH:PATH_TO_ADMLP
mpirun -np 1 python ADMLP/test.py
```

# Closed Loop Evaluation
1. Preparations:
- Clone Bench2Drive from [here](https://github.com/Thinklab-SJTU/Bench2Drive).
- Follow [this](https://github.com/Thinklab-SJTU/Bench2Drive/tree/main#setup) to install CARLA and python egg.

2. Link this repo to Bench2Drive
- Follow [Bench2Drive Official Repo](https://github.com/Thinklab-SJTU/Bench2Drive?tab=readme-ov-file#eval-tools) to install CARLA and CARLA python egg.
- link the two agents `admlp_b2d_agent_jittor.py` and `tcp_b2d_agent_jittor.py` under `leaderboard/team_code` folder of Bench2Drives
- Modify the script "run_evaluation_debug.sh" to configure the team code agent, model config, and model checkpoint to run.

3. Run Evaluation
Follow [this](https://github.com/Thinklab-SJTU/Bench2Drive?tab=readme-ov-file#eval-tools) to use evaluation tools of Bench2Drive.