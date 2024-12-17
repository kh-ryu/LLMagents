# LLMagents
Course project for CS294 LLM agents

### Demo video
<p align="center">
  <img src="./assets/multi_agent_small_room_with_caption.gif" width=100%>
</p>

## Installation

Create Conda Environment
```bash
conda create -n cs294_llmagents python=3.9
```
Install dependencies
```bash
pip install -r requirements.txt
```

## Experiment

You need to move to the [`multigrid`](./multigrid) directory to execute the experiments.
```bash
cd multigrid
```

#### Single agent
4x4 room size
```bash
python kanghyun_single_agent_test.py
```
8x8 room size
```bash
python kanghyun_single_agent_large_test.py
```

#### Multi agent
4x4 room size
```bash
python kanghyun_multiagent_test.py
```
8x8 room size
```bash
python kanghyun_multiagent_large_test.py
```

# References
For the experiment environment, we used the multigrid environment provided by the following repository.
### Multigrid environment

https://github.com/ini/multigrid/tree/master
