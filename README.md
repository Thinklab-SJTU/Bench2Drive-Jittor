
<h2 align="center">
  <img src="assets/jittor-bench2drive.png" style="width: 100%; height: auto;">
</h2>
<h2 align="center">
Bench2DriveZoo - Jittor
</h2>
<h2 align="center">
  <img src="assets/bench2drivezoo.png" style="width: 100%; height: auto;">
</h2>

# Introduction (介绍)

This repo contains the [Jittor](https://github.com/Jittor/jittor) implementation of [Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo), which supports [BEVFormer](https://github.com/fundamentalvision/BEVFormer), [UniAD](https://github.com/OpenDriveLab/UniAD) , [VAD](https://github.com/hustvl/VAD) in [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive). **All models are student models of the world model RL teacher - [Think2Drive](https://arxiv.org/abs/2402.16720).**

本仓库是[Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo)基于[计图](https://github.com/Jittor/jittor)国产深度学习框架的实现，支持在闭环端到端自动驾驶测试基准[Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)中运行。本项目在[JAD](https://github.com/Jittor/JAD)基础上，将[UniAD](https://github.com/OpenDriveLab/UniAD)和[VAD](https://github.com/hustvl/VAD)适配到CARLA仿真，使用基于世界模型的强化学习教师[Think2Drive](https://arxiv.org/abs/2402.16720)采集的数据进行模仿学习得到。


We also implement [AD-MLP](https://arxiv.org/abs/2305.10430) and [TCP](https://arxiv.org/abs/2206.08129) in Bench2Drive under Jittor. Use "git checkout tcp/admlp" to obtain their corresponding training and evaluation code.

我们也实现了Jiitor框架的[AD-MLP](https://arxiv.org/abs/2305.10430)和[TCP](https://arxiv.org/abs/2206.08129)在CARLA下的版本。请使用"git checkout tcp/admlp"切换到对应的分支。



# Prepare (配置环境 & 准备数据集)

- [Installation (环境配置)](docs/INSTALL.md)
- [Prepare Dataset (数据集准备)](docs/DATA_PREP.md)
- [Convert Codes from nuScenes to Bench2DriveZoo-Jittor (迁移模型到本仓库)](docs/CONVERT_GUIDE.md)

# Open-loop evaluation (开环验证)

1. Prepare your checkpoint, or download our [pretrained models](https://huggingface.co/rethinklab/Bench2DriveZoo/tree/main)
准备好自己的checkpoint 或者 使用[预训练好的模型]((https://huggingface.co/rethinklab/Bench2DriveZoo/tree/main))。


2. Open the root directory, run:
在本目录下运行：
```bash
bash ./adzoo/uniad(vad)/uniad(vad)_jittor_eval.sh ./adzoo/uniad(vad)/configs/.../your_config.py /path/to/xxx.pth 1

e.g.:
bash ./adzoo/uniad/uniad_jittor_eval.sh ./adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py ./ckpts/uniad_base_b2d.pth 1
bash ./adzoo/vad/vad_jittor_eval.sh ./adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py ./ckpts/vad_b2d_base.pth 1
```



# Close-loop evaluation (闭环评测)
- Follow [Bench2Drive Official Repo](https://github.com/Thinklab-SJTU/Bench2Drive?tab=readme-ov-file#eval-tools) to install CARLA and CARLA python egg.
- link the two agents `uniad_b2d_agent_jittor.py` and `vad_b2d_agent_jittor` under `leaderboard/team_code` folder of Bench2Drives
- Modify the script "run_evaluation_debug.sh" to configure the team code agent, model config, and model checkpoint to run.
- Open the root directory of **Bench2Drive**, then
```shell
bash ./leaderboard/scripts/run_evaluation_debug.sh
```


# Citation (引用) <a name="citation"></a>

Please consider citing the following papers if the project helps your research with the following BibTex:

如果您觉得本项目对您有帮助，请考虑引用：

```bibtex
@article{hu2020jittor,
  title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
  author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
  journal={Science China Information Sciences},
  volume={63},
  number={222103},
  pages={1--21},
  year={2020}
}

@article{jia2024bench,
  title={Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-To-End Autonomous Driving},
  author={Xiaosong Jia and Zhenjie Yang and Qifeng Li and Zhiyuan Zhang and Junchi Yan},
  journal={arXiv preprint arXiv:2406.03877},
  year={2024}
}

@inproceedings{li2024think,
  title={Think2Drive: Efficient Reinforcement Learning by Thinking in Latent World Model for Quasi-Realistic Autonomous Driving (in CARLA-v2)},
  author={Qifeng Li and Xiaosong Jia and Shaobo Wang and Junchi Yan},
  booktitle={ECCV},
  year={2024}
}
```