import gymnasium as gym
import multigrid.envs
import openai, os, re
import logging
from openai import OpenAI
from multigrid.core.actions import Moveonly_Action, Action
from prompt.utils import file_to_string
from multigrid.core.agent import Agent, MissionSpace



# log_file = "agent_log.log"

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.FileHandler(log_file),
#         logging.StreamHandler()        
#     ]
# )

class LLMAgent(Agent):
   def __init__(self, system_prompt, *args, **kwargs):
      super().__init__(*args, **kwargs)
    #   self.system_prompt = file_to_string(system_prompt)
    #   self.messages = [{"role": "system", "content": self.system_prompt}]
    #   self.llm = OpenAI()
      
   def response(self, obs):
      action = None
      messages = self.messages.copy()
      messages.append({"role": "user", "content": obs})
      self.messages.append({"role": "user", "content": obs})
      logging.info(f"Agent {self.index} Observation: {obs}")
      
      for i in range(3):
         response = self.llm.chat.completions.create(
            model='gpt-4o',
            messages=self.messages,
            max_tokens=1000,
            temperature=0.0,
            top_p=0.9
         ).choices[0].message.content
         
         messages.append({"role": "assistant", "content": response})
         logging.info(f"Agent {self.index} Response: {response}")
         pattern = r"Action:\s*([A-Za-z]+\(.*?\))"
         match = re.search(pattern, response)
         if match:
            action = match.group(1)
            action = ACTION_SPACE[action.lower().strip()]
            if action:
               break
            else:
               obs = "Failed to parse your action."
               continue
         else:
            obs = "Failed to parse your action."
            continue
      
      self.messages.append({"role": "assistant", "content": response})
      return action

system_prompt_folder = "maulik_prompt"
system_prompt = file_to_string(f"./prompt/{system_prompt_folder}/system_prompt_multi.txt")

print(system_prompt)

client = OpenAI()
messages = [{"role": "system", "content": system_prompt}]

ACTION_SPACE = {
    "left()": Action.left,
    "right()": Action.right,
    "forward()": Action.forward,
    "pickup()": Action.pickup,
    "drop()": Action.drop,
    "toggle()": Action.toggle,
    "done()": Action.done
}
      
mission = MissionSpace.from_string("Pick up the goal")
agents = [LLMAgent(system_prompt=system_prompt, index=0, mission_space=mission, view_size=3), 
          LLMAgent(system_prompt=system_prompt,index=1, mission_space=mission, view_size=3)]

env = gym.make('MultiGrid-BlockedUnlockPickup-v0', agents=agents, render_mode='human')
env = env.unwrapped

obs = {}

obs[0] = "Let's think step by step. Both agents are on a grid now; please begin exploring the environment to gather some information about it."
obs[1] = "Let's think step by step. Both agents are on a grid now; please begin exploring the environment to gather some information about it."

# 
# for agent in agents:
#    obs[agent.index] = "Observation: You are begining to solve the task now. Please explore the environment and try to pick up the goal."

t = 0

observations, infos = env.reset()
done = False
while not done:

    if t == 0:
        actions = {0: ACTION_SPACE["forward()"], 1: ACTION_SPACE["forward()"]}
        t += 1

    else:

        print("Agent-1: "+obs[0]+"Agent-2: "+obs[1])

        messages.append({"role": "user", "content": "Agent-1: "+obs[0]+"Agent-2: "+obs[1]})

        # print(f"Observation: {obs}")
        # print(messages)
        response = client.chat.completions.create(
            model='gpt-4o',
            messages=messages,
            max_tokens=1000,
            temperature=0.0,
            top_p=0.9
        ).choices[0].message.content
        messages.append({"role": "assistant", "content": response})
        print(f"Assistant: {response}")
        pattern1 = r"Action-1:\s*([A-Za-z]+\(.*?\))"
        match1 = re.search(pattern1, response)
        pattern2 = r"Action-2:\s*([A-Za-z]+\(.*?\))"
        match2 = re.search(pattern2, response)
        actions = {}
        if match1:
            action = match1.group(1)
            actions[0] = ACTION_SPACE[action.lower().strip()]
        else:
            obs[0] = "Failed to parse your action."
            continue
        if match2:
            action = match2.group(1)
            actions[1] = ACTION_SPACE[action.lower().strip()]
        else:
            obs = "Failed to parse your action."
            continue

    observations, rewards, terminations, truncations, infos = env.step(actions)
    for agent_index, observation in observations.items():
        obs[agent_index] = observation["text"][0] if isinstance(observation["text"], list) else observation["text"]

    done = any([rewards[idx] for idx in rewards.keys()])
   
env.close()
