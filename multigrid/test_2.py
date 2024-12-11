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
   - Note that you are receiving a first-person perspective of the grid, with you always centered in the bottom-middle cell of the grid. When following python indexing, the bottom-left cell is (0, 0). So, your current position is always (1,0) in the grid.
   - After each action, you receive a grid representation with symbols that describe the immediate environment:
	- You: Your current position.
   - Empty : An Empty cell, accessible for movement.
   - Wall : which blocks movement.
   - Door : which may require toggling to pass through.
   - Key : an interactable item which will open the door.
   - Ball : an interactable item.
   - Box : an interactable item.
   - Goal : indicating a key destination.
   - Lava : a dangerous cell that should be avoided.
   - Unseen: area outside your current visibility, potentially beyond the map's boundaries or obscured by walls or other inaccessible obstacles.
   - @: Another agent present in the environment. They are your teammates, and you work together to complete the assigned tasks.

# Action Space #
You can choose from the following actions to interact with the environment:
	- Left(): Turn left.
	- Right(): Turn right.
	- Forward(): Move forward.
	- Pickup(): Picks up a ball, key, or goal if it is in the position (1,1) in the grid which is right in front of you.
	- Drop(): Drop the currently held object.
	- Toggle(): Toggle or activate an object (e.g., a door).
	- Done(): Indicate that the task is complete.
## Movement Rules:
    - You can only move to adjacent cells marked as Empty.
    - Movement into walls, lava, or unseen areas is not allowed.
    - Plan your actions strategically to avoid backtracking and maximize exploration efficiency.
    - If you can't find a desired object, try using different actions to explore the environment and locate it.
    - You can only pick up an object when it is in the position (1,1) in the grid which is right in front of you. If you try to pick up or toggle and the object is not in the position (1,1) in the grid, the action will not be successful.
    - Strictly don't try the same action again and again if it is not successful. Try to explore the environment and find the object in the right position.

# Response Format #
Your responses must follow this format:
   - Thought: Provide your reasoning based on the current grid and describe your understanding of the environment. This should include observations, updates to your internal map, and predictions for future actions.(Prefix: Thought: ...)
	- Action: Specify the next action you will take, strictly chosen from the available action space.(Prefix: Action: ...)

# Objective #
- Your goal is to open a door which is locked. 
- Right in front of the door, there is a ball that is blocking the door. 
- You need to move to a location which is just one move away from the ball. This means that in your observation grid ball should be in the position (1,1) in the grid.
- pick the up the ball and move to a different location that does not block the door and drop the ball there. 
- Do not pick up the ball once you have dropped it.
- Only after you are done with all the previous tasks, you should try and find the key.
- Same as before, move to a location such that key is in position (1,1) in your observation grid and pick up the key
- Try to explore the environment by using forward() and left() or right() actions to find the door.
- go to the location adjecent to the door such that door is directly in front of you in your observation grid and open the door.
- Go to the goal location in the other room to complete the task. Remeber to use action done() only after the goal is in position (0,1) in your observation grid.
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

obs = "Let's think step by step. You are on a grid now; please begin exploring the environment to gather some information about it."


observations, infos = env.reset()
done = False
while not done:
   
   # messages.append({"role": "user", "content": obs})
   messages.append({"role": "user", "content": obs})

   # print(messages)
   print(f"Observation: {obs}")
   response = client.chat.completions.create(
      model='gpt-4o',
      messages=messages,
      max_tokens=1000,
      temperature=0.3,
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

   messages.pop()
   


env.close()


"""" 

# Example #
## Example 1
Observation:
[[░, ░, ░],
 [░, █,   ],
 [░, ^, ★]]

Response:
Thought: From my perspective, there is a wall (`█`) directly in front of me, blocking my path. To my left is an unseen area (`░`), meaning it is currently outside my field of view. However, I can see that the goal (`★`) is located to my right. Since the goal is within my visible range and accessible, the most logical action is to turn right to align myself with it and move closer to completing the task.
Action: Right()

 ## Example 2
Observation:
[[░, ░, ░],
 [░, ░, '● (Color: green, State: open)'],
 [░, ^, ░]]

Response:
Thought: I see a ball (`●`) in front of me, but it is not in the right position for me to pick it up. I should move forward to position myself correctly before attempting to pick it up.
Action: Forward()

## Example 3
Observation:
[[░, ░, ░],
 [░, ░, ░],
 [░, ^, '● (Color: green, State: open)']]

Response:
Thought: The ball (`●`) is now to right of me so I should turn right to position myself correctly.
Action: Right()

## Example 4
Observation:
[[░, ░, ░],
 [░, '● (Color: green, State: open)', ░],
 [░, ^, ░]]

Response:
Thought: The ball (`●`) is now in direct in front of me. Which is correct position for me to pick it up. I should do so before proceeding further.
Action: Pickup()

## Example 5
Observation:
[[░, ░, ░],
 ['† (Color: purple, State: open)', ░, ░],
 [░, ^, ░]]

Response:
Thought: I see a key (`†`) in front of me, but it is not in the right position for me to pick it up. I should move forward to position myself correctly before attempting to pick it up.
Action: Forward()

## Example 6
Observation:
[[░, ░, ░],
 [░, ░, ░],
 ['† (Color: purple, State: open)', ^, ░]]

Response:
Thought: The key (`†`) is now to left of me so I should turn left to position myself correctly.
Action: Left()

## Example 7
Observation:
[[░, ░, ░],
 ['† (Color: purple, State: open)', ░, ░],
 [░, ^, ░]]

Response:
Thought: The key (`†`) is now in direct in front of me. Which is correct position for me to pick it up. I should do so before proceeding further.
Action: Pickup()


# # Movement Suggestions #
- If the desired object in the positon (2,0), you can turn right and the object will be in the position (1,1) in the grid.
- If the desired object in the positon (0,0), you can turn left and the object will be in the position (1,1) in the grid.
- If the desired object is in the position (0,1), you can move forward and then move left and the object will be in the position (1,1) in the grid.
- If the desired object is in the position (2,1), you can move forward and then move right and the object will be in the position (1,1) in the grid.
 """