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

mission1 = """
Your current goal is to pick up the ball blocking the door and drop at the any other place.
After you pick up the ball, drop it to any place that is not an original place that blocks the door.
Remember, you can only move to front cells marked as Empty.
Movement into walls, lava, or unseen areas is not allowed.
Remember, you can only pickup objects that are directly **in front of** you.
Let’s think step by step, and provide one moveable action first.
"""

mission2 = """
Your current goal is to pick up the key and open the door with that key.
First, find the key and pick up the key by using pick up action when the key is in front of you.
Then, open the door by toggle action when you are facing the door in front of you.
Do not pick up the ball again as ball is moved to desired place.
Do not worry about losing the door in the sight. Some turning behavior might necessary to go aroun the object blocking the door.
Remember that you should carry the key when you open the door.
Remember, you can only move to the cell in front of you marked as Empty.
Movement into walls, lava, or unseen areas is not allowed.
Remember, you can only pickup or toggle objects that are directly **in front of** you.
Let’s think step by step, and provide one moveable action first.
"""



agents = [
   LLMAgent(color="blue", index=0, mission_str=mission1,view_size=3, restricted_obj=["key"], sys_prompt_path="prompt/kanghyun_prompt/multi_system_prompt.txt", availiable_action=availiable_action), 
   LLMAgent(color="green", index=1, mission_str=mission2, view_size=3, restricted_obj=["ball"],  sys_prompt_path="prompt/kanghyun_prompt/multi_system_prompt.txt", availiable_action=availiable_action)]

env = gym.make('MultiGrid-BlockedUnlockPickup-v0', agents=agents, render_mode='human')
env = env.unwrapped
multiagent = MultiAgent(env, agents)
obs = {0: file_to_string("prompt/kanghyun_prompt/user_prompt_ball.txt"), 1: file_to_string("prompt/kanghyun_prompt/user_prompt_key.txt")}

with open("./result.txt","w") as f:
   pass

for i in range(10):
   env.reset()
   result, step = multiagent.solve(obs, max_steps=150)
   with open("./result.txt","a") as f:
      f.write(f"Epoch {i + 1}, step {step}, Success {result}\n")
   