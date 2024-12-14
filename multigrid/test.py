import gymnasium as gym
import multigrid.envs
import openai, os, re
import logging
from openai import OpenAI
from multigrid.multiagent.multiagent import MultiAgent
from multigrid.multiagent.action import *
from multigrid.multiagent.llmagent import LLMAgent
from prompt.utils import file_to_string

availiable_action = [Forward, Left, Right, Pickup, Drop, Toggle, Done]

agents = [
   LLMAgent(color = "blue", index=0, view_size=3, restricted_obj=["key"], sys_prompt_path="prompt/kanghyun_prompt/system_prompt.txt", availiable_action=availiable_action), 
   LLMAgent( color="green", index=1, view_size=3, restricted_obj=["ball"],  sys_prompt_path="prompt/kanghyun_prompt/system_prompt.txt", availiable_action=availiable_action)]

env = gym.make('MultiGrid-BlockedUnlockPickup-v0', agents=agents, render_mode='human')
env = env.unwrapped
multiagent = MultiAgent(env, agents)
obs = {0: file_to_string("prompt/kanghyun_prompt/user_prompt_ball.txt"), 1: file_to_string("prompt/kanghyun_prompt/user_prompt_key.txt")}


multiagent.solve(obs)