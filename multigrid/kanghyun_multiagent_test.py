import gymnasium as gym
import multigrid.envs
import openai, os, re
import logging
from openai import OpenAI
from multigrid.core.actions import Moveonly_Action, Action
from prompt.utils import file_to_string, gpt_interaction
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

system_prompt_folder = "multi_agent_prompt"
key_agent_system_prompt = f"./prompt/{system_prompt_folder}/key_agent_system.txt"
ball_agent_system_prompt = f"./prompt/{system_prompt_folder}/ball_agent_system.txt"
key_agent_user_prompt = f"./prompt/{system_prompt_folder}/key_agent_user.txt"
ball_agent_user_prompt = f"./prompt/{system_prompt_folder}/ball_agent_user.txt"

ACTION_SPACE_KEY = {
   "left()": Action.left,
   "right()": Action.right,
   "forward()": Action.forward,
   "pickup()": Action.pickup,
   "drop()": Action.drop,
   "toggle()": Action.toggle,
   "done()": Action.done
}

ACTION_SPACE_BALL = {
   "left()": Action.left,
   "right()": Action.right,
   "forward()": Action.forward,
   "pickup()": Action.pickup,
   "drop()": Action.drop,
   "toggle()": Action.toggle,
   "done()": Action.done
}

openai.api_key = os.getenv('OPENAI_API_KEY')

class LLMAgent(Agent):
   def __init__(self, role, *args, **kwargs):
      super().__init__(*args, **kwargs)
   
      self.llm = OpenAI()
      if role == "key":
         self.action = ACTION_SPACE_KEY
         self.system_prompt = file_to_string(key_agent_system_prompt)
         self.user_prompt = file_to_string(key_agent_user_prompt)
      elif role == "ball":
         self.action = ACTION_SPACE_BALL
         self.system_prompt = file_to_string(ball_agent_system_prompt)
         self.user_prompt = file_to_string(ball_agent_user_prompt)
      
      self.messages = [{"role": "system", "content": self.system_prompt},
                       {"role": "user", "content": self.user_prompt}]
      
      self.history = []
      
   def response(self, obs):
      action = None
      self.history.append({"role": "user", "content": obs})
      # logging.info(f"Agent {self.index} Observation: {obs}")
      print(f"Agent {self.index} Color: ", self.color)
      print(f"Agent {self.index} Observation: ", obs)
      
      for i in range(3):
         response = gpt_interaction(self.llm, "gpt-4o", self.messages + self.history)
         print(f"Agent {self.index} Response: ", response)
         # logging.info(f"Agent {self.index} Response: {response}")
         pattern = r"Action:\s*([A-Za-z]+\(.*?\))"
         match = re.search(pattern, response)
         if match:
            action = match.group(1)
            try:
               action = self.action[action.lower().strip()]
               break
            except:
               action = None
               obs = "Failed to parse your action."
               continue
         else:
            obs = "Failed to parse your action."
            continue
      
      self.history.append({"role": "assistant", "content": response})
      self.history = self.history[-5:]
      return action
   
   def talk(self, messages):
      pass
 
mission = MissionSpace.from_string("Pick up the goal")
agents = [LLMAgent(role='ball', color = "blue", index=0, mission_space=mission, view_size=3, restricted_obj=["key"]), 
          LLMAgent(role='key', color="green", index=1, mission_space=mission, view_size=3, restricted_obj=["ball"])]

env = gym.make('MultiGrid-BlockedUnlockPickup-v0', agents=agents, render_mode='human')
env = env.unwrapped

observations, infos = env.reset()
text_obs = {}
for agent_index, observation in observations.items():
   text_obs[agent_index] = observation["text"][0] if isinstance(observation["text"], list) else observation["text"]

done = False
while not done:
   actions = {}

   for agent in agents:
      action = agent.response(text_obs[agent.index])
      if action is None:
         actions[agent.index] = Action.done
      else:
         actions[agent.index] = action
   
   observations, rewards, terminations, truncations, infos = env.step(actions)
   for agent_index, observation in observations.items():
      text_obs[agent_index] = observation["text"][0] if isinstance(observation["text"], list) else observation["text"]
      
   done = any([rewards[idx] for idx in rewards.keys()])
   
env.close()

