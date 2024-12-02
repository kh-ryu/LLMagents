import gymnasium as gym
import multigrid.envs
import openai, os, re
from openai import OpenAI
from multigrid.core.actions import Moveonly_Action, Action


openai.api_key = os.getenv('OPENAI_API_KEY')

system_prompt = """
You are a helpful assistant skilled in navigating grid environments and completing item pickup tasks.

Your vision is limited to a specific area of the map, meaning you cannot see the entire environment at once. As a result, you need to explore the environment on your own to gather information.

Your response format should be:
Thought: ...
Action: ...
Where Thought includes your reasoning about the current environment and prediction of the next action, and Action specifies one action from the Action space.

Available Actions:
- Left(): Turn left
- Right(): Turn right
- Forward(): Move forward
- Pickup(): Pick up an object
- Drop(): Drop an object
- Toggle(): Toggle or activate an object
- Done(): Done completing the task

Key Details:

1.	Feedback on Movement:
Each time you move, you will receive feedback in the form of a textual grid representation that shows information about your surrounding environment:
	- You: Indicates your current position.
	- Empty: Indicates a grid position that is empty and accessible.
	- Unseen: Indicates grid positions that are beyond your current visibility and are unknown.
	- Wall: Indicates a wall that blocks movement.
2.	Movement Rules:
	- You can only move to adjacent grid cells marked as Empty.
	- You must carefully analyze your surroundings to avoid unnecessary backtracking and maintain an efficient exploration strategy.

Objective:
Your goal is to complete the assigned task as faithfully and efficiently as possible, while ensuring your actions are safe and logical. Use the feedback provided after each move to update your understanding of the environment and plan your next steps accordingly..
"""

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



# import pdb
# pdb.set_trace()

obs = "Let's think step by step. You are at a grid now, please begin to explore the environment and solve the task"


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
      action = [ACTION_SPACE[action.lower().strip()]]
   else:
      obs = "Failed to parse your action."
      continue


   observations, rewards, terminations, truncations, infos = env.step(action)
   env.render()

   obs = observations[0]["text"][0] if isinstance(observations[0]["text"], list) else observations[0]["text"]
   done = any(terminations.values())

env.close()