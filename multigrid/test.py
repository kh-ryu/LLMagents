import gymnasium as gym
import multigrid.envs
import openai, os, re
from openai import OpenAI
from multigrid.core.actions import Moveonly_Action, Action
from prompt.utils import file_to_string

openai.api_key = os.getenv('OPENAI_API_KEY')

system_prompt_folder = "maulik_prompt"
user_prompt_folder = "kanghyun_prompt"

system_prompt = file_to_string(f"./prompt/{system_prompt_folder}/system_prompt.txt")

client = OpenAI()
messages = [{"role": "system", "content": system_prompt}]
env = gym.make('MultiGrid-BlockedUnlockPickup-v0', agents=1, render_mode='human')
env = env.unwrapped

ACTION_SPACE = {
    "left()": Action.left,
    "right()": Action.right,
    "forward()": Action.forward,
    "pickup()": Action.pickup,
    "drop()": Action.drop,
    "toggle()": Action.toggle,
    "done()": Action.done
}

obs = file_to_string(f"./prompt/{user_prompt_folder}/user_prompt.txt")

observations, infos = env.reset()
done = False
while not done:
   messages.append({"role": "user", "content": obs})
   print(f"Observation: {obs}")
   response = client.chat.completions.create(
      model='gpt-4o',
      messages=messages,
      max_tokens=1000,
      temperature=0.0,
      top_p=0.9
   ).choices[0].message.content
   messages.append({"role": "assistant", "content": response})
   print(f"Assistant: {response}")
   pattern = r"Action:\s*([A-Za-z]+\(.*?\))"
   match = re.search(pattern, response)
   if match:
      action = match.group(1)
      action = {0: ACTION_SPACE[action.lower().strip()]}
   else:
      obs = "Failed to parse your action."
      continue
   
   observations, rewards, terminations, truncations, infos = env.step(action)
   
   obs = observations[0]["text"][0] if isinstance(observations[0]["text"], list) else observations[0]["text"]
   done = any(terminations.values())
   
env.close()
