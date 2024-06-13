# insight
Official implementation of the paper "End-to-End Neuro-Symbolic Reinforcement Learning with Textual Explanations" (ICML 2024)
#
<div align="center">

# End-to-End Neuro-Symbolic Reinforcement Learning with Textual Explanations

### ICML 2024 (Spotlight)

**[Lirui Luo](https://liruiluo.github.io/), [Guoxi Zhang](https://guoxizhang.com/), [Hongming Xu](), [Yaodong Yang](https://www.yangyaodong.com/), [Cong Fang](https://congfang-ml.github.io/), [Qing li](https://liqing-ustc.github.io/)**


| [```Website```](https://liruiluo.github.io/nsrl-vision-pub/) | [```Arxiv```](https://arxiv.org/abs/2403.12451) |
:------------------------------------------------------:|:-----------------------------------------------:|

<img src="docs/figures/teaser-1.png" width="568">

</div>

---

# Abstract

Neuro-symbolic reinforcement learning (NS-RL) has emerged as a promising paradigm for explainable decision-making, characterized by the interpretability of symbolic policies.
NS-RL entails structured state representations for tasks with visual observations, but previous methods are unable to refine the structured states with rewards due to a lack of efficiency.
Accessibility also remains to be an issue, as extensive domain knowledge is required to interpret symbolic policies.
In this paper, we present a framework for learning structured states and symbolic policies jointly, whose key idea is to distill vision foundation models into a scalable perception module and refines it during policy learning.
Moreover, we design a pipeline to generate language explanations for policies and decisions using large language models.
In experiments on nine Atari tasks, we verify the efficacy of our approach, and we also present explanations for policies and decisions.

<div align="center">
<table>
<tr>
<td>
<img src="docs/figures/ICML-Framework-1.png" >
</td>
</tr>
<tr>
<th>
The INSIGHT framework.
</th>
</tr>
</table>
</div>

# Results


Here is the segmentation videos before and after policy learing on Freeway:

<div align="center">
<table>
<tr>
<td>
<img src="docs/gifs/Freeway_before264.gif" controls>
</img>

</td>
</tr>
<tr>
<th>
The video before.
</th>
</tr>
</table>
</div>

<div align="center">
<table>
<tr>
<td>
<img src="docs/gifs/Freeway264.gif" controls>
</img>
</td>
</tr>
<tr>
<th>
The video after.
</th>
</tr>
</table>
</div>

---

# Usage

## Installation
Prerequisites:
* Python==3.9.17

```bash
# core dependencies
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-atari.txt
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
pip install git+https://github.com/DLR-RM/stable-baselines3
git clone https://github.com/metadriverse/metadrive.git
cd metadrive
pip install -e .
pip install -e .[cuda]
conda install -c nvidia cuda-python
cd ..
cd cleanrl
cd sam_track
bash script/install.sh
bash script/download_ckpt.sh
cd FastSAM
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```


## Download Necessary Ckpts to cleanrl/sam_track/ckpt

[Track Model](https://drive.google.com/file/d/1g4E-F0RPOx9Nd6J7tU9AE1TjsouL4oZq/view)

[FastSAM](https://drive.google.com/file/d/1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv/view)

## Train INSIGHT

To generate dataset, use
```bash
cd ..
python demo.py --video-name PongNoFrameskip-v4
```
To train cnn, use
```bash
cd ..
python train_cnn.py --wandb-project-name nsrl-eql --env-id PongNoFrameskip-v4 --run-name benchmark-pretrain-Pong-seed1 --seed 1
```
Or you can use a build-in [dataset](https://drive.google.com/file/d/1E_b3eBJ47ze1OJ7Nz1khsJ-q1YrcjTdu/view?usp=sharing) directly

To train policy, use 
```bash
python train_policy_atari.py --wandb-project-name nsrl-eql --env-id PongNoFrameskip-v4 --run-name benchmark-ng-reg-weight-1e-3-Pong-seed1 --ng True --reg_weight 1e-3 --seed 1 --load_cnn True
```

To train metadrive, use 
```bash
python train_policy_metadrive.py --wandb-project-name nsrl-eql --run-name benchmark-INSIGHT-MetaDriveEnv-seed1 --env-id MetaDriveEnv --cnn_loss_weight 2 --distillation_loss_weight 1 --load_cnn True --seed 1 --learning-rate 5e-5 --clip-coef 0.2 --ent-coef 0.01 --ego_state True --num-envs 8 --num-steps 125 --update-epochs 4 --num-minibatches 10 --max-grad-norm 0.5 --anneal-lr False --kl-penalty-coef 0.2 --reg_weight 1e-4  --use_eql_actor True
```


# Citation

If you find our code implementation helpful for your own research or work, please cite our paper.

```bibtex
@article{luo2024insight,
  title={End-to-End Neuro-Symbolic Reinforcement Learning with Textual Explanations},
  author={Luo, Lirui and Zhang, Guoxi and Xu, Hongming and Yang, Yaodong and Fang, Cong and Li, Qing},
  journal={ICML},
  year={2024}
}
```

# Contact

For any queries, please [raise an issue](https://github.com/liruiluo/nsrl-vision-pub/issues/new) or
contact [Qing Li](https://liqing-ustc.github.io/).

# License

This project is open sourced under [MIT License](LICENSE).
