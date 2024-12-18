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
Your current goal consists of three tasks, with Task 1 and Task 2 being interchangeable based on the actual environment. Proceed in the most efficient order depending on the current situation.
Task 1: Move the Ball Blocking the Door
	- Pick up the ball blocking the door.
	- Drop the ball at any place that is not the original position blocking the door.
	- Ensure the ball is not placed in front of the door again.
Task 2: Pick up the Key and Open the Door
	- Find the key and pick it up using the pick-up action when the key is directly in front of you.
	- Carry the key while navigating toward the door.
	- Use the toggle action to open the door when you are directly facing it.
	- Do not pick up the ball again, as it has already been moved to a suitable place.
Task 3: Pick up the Goal in Another Room
	- After opening the door, navigate to the next room.
	- Find the Goal object located in that room.
	- Pick up the Goal using the pick-up action when it is directly in front of you.
"""



agents = [
   LLMAgent(color="blue", index=0, mission_str=mission1,view_size=3, restricted_obj=[], sys_prompt_path="prompt/kanghyun_prompt/multi_system_prompt.txt", availiable_action=availiable_action), 
   LLMAgent(color="green", index=1, mission_str=mission1, view_size=3, restricted_obj=[],  sys_prompt_path="prompt/kanghyun_prompt/multi_system_prompt.txt", availiable_action=availiable_action)]

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
   