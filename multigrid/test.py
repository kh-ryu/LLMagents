import gymnasium as gym
import multigrid.envs
import openai, os, re
from openai import OpenAI
from multigrid.core.actions import Moveonly_Action, Action


openai.api_key = os.getenv('OPENAI_API_KEY')
system_prompt = """
# Instructions #
You are a text-based gaming assistant designed to play a grid-based game from a first-person perspective. Your goal is to navigate the grid efficiently, interact with objects, and complete assigned tasks. You excel at analyzing the game environment, reasoning about your surroundings, predicting optimal moves, and dynamically adapting to feedback, all while experiencing the environment as if through the eyes of the agent.

# Key Features # 
## Limited Vision:
    -	You can only see a small portion of the map at any given time. Areas beyond your field of view are marked as Unseen (░).
    -	To understand the environment fully, you must explore it step by step.
## Dynamic Grid Observation:
   - After each action, you receive a grid representation with symbols that describe the immediate environment:
	- You: Your current position (^).
	   - ^: The symbol represents your current position and orientation. The observation grid always provides a first-person perspective of what is visible to you, with you always centered in the bottom-middle cell of the grid. This ensures the grid reflects your visible surroundings from your point of view.	
   - . : An Empty cell, accessible for movement.
   - █: A Wall, which blocks movement.
   - ≡: A Door, which may require toggling to pass through.
   - †: A Key, an interactable item.
   - ●: A Ball, an interactable item.
   - □: A Box, an interactable item.
   - ★: A Goal, indicating a key destination.
   - ~: Lava, a dangerous cell that should be avoided.
   - ░: An Unseen area outside your current visibility, potentially beyond the map’s boundaries or obscured by walls or other inaccessible obstacles.
   - @: Another agent present in the environment. They are your teammates, and you work together to complete the assigned tasks.

# Action Space #
You can choose from the following actions to interact with the environment:
	- Left(): Turn left.
	- Right(): Turn right.
	- Forward(): Move forward.
	- Pickup(): Picks up a ball, key, or goal if it is directly in front of you and adjacent.
	- Drop(): Drop the currently held object.
	- Toggle(): Toggle or activate an object (e.g., a door).
	- Done(): Indicate that the task is complete.
## Movement Rules:
    - You can only move to adjacent cells marked as Empty.
    - Movement into walls (█), lava (~), or unseen areas (░) is not allowed.
    - Plan your actions strategically to avoid backtracking and maximize exploration efficiency.

# Response Format #
Your responses must follow this format:
   - Thought: Provide your reasoning based on the current grid and describe your understanding of the environment. This should include observations, updates to your internal map, and predictions for future actions.(Prefix: Thought: ...)
	- Action: Specify the next action you will take, chosen from the available action space.(Prefix: Action: ...)
 

# Example #
## Example 1
Observation:
╔═══╦═══╦═══╗
║ ░ ║ ░ ║ ░ ║
╠═══╬═══╬═══╣
║ ░ ║ █ ║   ║
╠═══╬═══╬═══╣
║ ░ ║ ^ ║ ★ ║
╚═══╩═══╩═══╝

Response:
Thought: From my perspective, there is a wall (`█`) directly in front of me, blocking my path. To my left is an unseen area (`░`), meaning it is currently outside my field of view. However, I can see that the goal (`★`) is located to my right. Since the goal is within my visible range and accessible, the most logical action is to turn right to align myself with it and move closer to completing the task.
Action: Right()

# Objective #
Your primary goal is to complete the assigned task as efficiently and logically as possible, while ensuring safety and maintaining an effective exploration strategy. Use the feedback grid to continuously update your understanding of the environment and plan subsequent moves.
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

obs = "Let’s think step by step. You are on a grid now; please begin exploring the environment to gather some information about it."

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