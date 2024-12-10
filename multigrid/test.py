import gymnasium as gym
import multigrid.envs
import openai, os, re
import logging
from openai import OpenAI
from multigrid.core.actions import Moveonly_Action, Action
from prompt.utils import file_to_string
from multigrid.core.agent import Agent, MissionSpace



log_file = "agent_log.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()        
    ]
)

system_prompt_folder = "kanghyun_prompt"
system_prompt = f"./prompt/{system_prompt_folder}/system_prompt.txt"

ACTION_SPACE = {
    "left()": Action.left,
    "right()": Action.right,
    "forward()": Action.forward,
    "pickup()": Action.pickup,
    "drop()": Action.drop,
    "toggle()": Action.toggle,
    "done()": Action.done
}


class LLMAgent(Agent):
   def __init__(self, system_prompt, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.system_prompt = file_to_string(system_prompt)
      self.messages = [{"role": "system", "content": self.system_prompt}]
      self.llm = OpenAI()
      
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
         if not action or not match:
            break
         else:
            obs = "Failed to parse your action."
            continue
        
      self.messages.append({"role": "assistant", "content": response})
      return action
         
      
mission = MissionSpace.from_string("Pick up the goal")
agents = [LLMAgent(system_prompt=system_prompt, index=0, mission_space=mission, view_size=3), 
          LLMAgent(system_prompt=system_prompt,index=1, mission_space=mission, view_size=3)]

env = gym.make('MultiGrid-BlockedUnlockPickup-v0', agents=agents, render_mode='human')
env = env.unwrapped

obs = {}
for agent in agents:
   obs[agent.index] = "Observation: You are begining to solve the task now. Please explore the environment and try to pick up the goal."

observations, infos = env.reset()
done = False
while not done:
   actions = {}

   for agent in agents:
      action = agent.response(obs[agent.index])
      assert action
      actions[agent.index] = action
   
   observations, rewards, terminations, truncations, infos = env.step(actions)
   for agent_index, observation in observations.items():
      obs[agent_index] = observation["text"][0] if isinstance(observation["text"], list) else observation["text"]
      
   

   done = any([rewards[idx] for idx in rewards.keys()])
   
env.close()
