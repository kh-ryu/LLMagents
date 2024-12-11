import gymnasium as gym
import multigrid.envs
import openai, os, re
from openai import OpenAI
from multigrid.core.actions import Moveonly_Action, Action
from autogen import ConversableAgent

def get_action(states):
    

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
	- You: Your current position (^).
	   - ^: The symbol represents your current position and orientation. The observation grid always provides a first-person perspective of what is visible to you, with you always centered in the bottom-middle cell of the grid. This ensures the grid reflects your visible surroundings from your point of view.	
   - . : An Empty cell, accessible for movement.
   - █: A Wall, which blocks movement.
   - ≡: A Door, which may require toggling to pass through.
   - †: A Key, an interactable item.
   - ●: A Ball, an interactable item.
   - □: A Box, the goal. You must reach this location to complete the task.
   - ~: Lava, a dangerous cell that should be avoided.
   - ░: An Unseen area outside your current visibility, potentially beyond the map's boundaries or obscured by walls or other inaccessible obstacles.
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
    - Movement into walls (█), lava (~), or unseen areas (░) is not allowed.
    - Plan your actions strategically to avoid backtracking and maximize exploration efficiency.

# Response Format #
Your responses must follow this format:
   - Thought: Provide your reasoning based on the current grid and describe your understanding of the environment. This should include observations, updates to your internal map, and predictions for future actions.(Prefix: Thought: ...)
	- Action: Specify the next action you will take, chosen from the available action space.(Prefix: Action: ...)
 

# Example #
## Example 1
Observation:
[[░, ░, ░],
 [░, █,   ],
 [░, ^, ★]]

Response:
Thought: From my perspective, there is a wall (█) directly in front of me, blocking my path. To my left is an unseen area (░), meaning it is currently outside my field of view. However, I can see that the goal (★) is located to my right. Since the goal is within my visible range and accessible, the most logical action is to turn right to align myself with it and move closer to completing the task.
Action: Right()

# Objective #
- Your goal is to open a door which is locked. 
- The door is blocked by a ball which must be moved before the door can be unlocked.
- You must move to the location next to the ball and pick it up.
- After picking up the ball, you need to move to a location that does not block the door and drop the ball there.
- If you did this once, you must not pick up and move the ball again.
- Only after you are done with all the previous tasks, you should try and find the key.
- You must move to the location where the key is present and pick it up.
- After that, you must move to the door and toggle it to unlock it.
- Go to the goal location in the other room to complete the task. Remmeber to use action done() only after the goal is in position (0,1) in your observation grid.
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

# entrypoint_agent_system_message = "You need to fetch the restaurant data for a given restaurant name. Call fetch_restaurant_data. Do not call calculate_overall_score until receiving a request. The restaurant name does not have to be exact matched, for instance, taco bell is treated as Taco Bell, In N Out is treated as In-n-Out, and McDonald is treated as McDonald's." # TODO
# example LLM config for the entrypoint agent
llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get('OPENAI_API_KEY')}]}
# the main entrypoint/supervisor agent
entrypoint_agent = ConversableAgent("entrypoint_agent", 
                                    system_message=system_prompt, 
                                    llm_config=llm_config)
# entrypoint_agent.register_for_llm(name="fetch_restaurant_data", description="Fetches the reviews for a specific restaurant.", api_style="function")(fetch_restaurant_data)
# entrypoint_agent.register_for_execution(name="fetch_restaurant_data")(fetch_restaurant_data)

# TODO
# Create more agents here. 

review_analyzing_agent_system_message = "You need to calculate the score for a restaurant based on the keywords in the reviews. Store the scores in two lists." # TODO
review_analyzing_agent = ConversableAgent("review_analyzing_agent", 
                                    system_message=review_analyzing_agent_system_message, 
                                    llm_config=llm_config)

scoring_agent_system_message = "You need to call a function calculate_overall_score." # TODO
scoring_agent = ConversableAgent("scoring_agent", 
                                    system_message=scoring_agent_system_message, 
                                    llm_config=llm_config)
entrypoint_agent.register_for_llm(name="calculate_overall_score", description="Calcualte the overall score of a restaurant.", api_style="function")(calculate_overall_score)
scoring_agent.register_for_execution(name="calculate_overall_score")(calculate_overall_score)

# TODO
# Fill in the argument to `initiate_chats` below, calling the correct agents sequentially.
# If you decide to use another conversation pattern, feel free to disregard this code.

# chat_results = entrypoint_agent.initiate_chats(
#    [
#       {
#             "recipient": entrypoint_agent,
#             "message": user_query + "You need to fetch the restaurant data for a given restaurant name and execute the function fetch_restaurant_data with the restaurant name as the argument. If each word of the given restaurant name starts with a small letter, you need to modify it to the corresponding capital letter. You just output the output of the function. Do not summary or modify your output from the output of the function. Do not call calculate_overall_score until receiving a request.",
#             "max_turns": 2,
#             "summary_method": "last_msg",
#       },
#       {
#             "recipient": review_analyzing_agent,
#             "message": "These are the restaurant name and reviews." + review_analyzer_prompt(),
#             "max_turns": 1,
#             "summary_method": "last_msg",
#       },
#       {
#             "recipient": scoring_agent,
#             "message": "These are the scores for each review. Call calculate_overall_score." + "calculate_overall_score('restaurant_name', [food_scores], [customer_service_scores])",
#             "max_turns": 2,
#             "summary_method": "last_msg",
#       },
#    ]
# )

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
   
   chat_results = entrypoint_agent.initiate_chats(
   [
      {
            "recipient": entrypoint_agent,
            "message": user_query + "You need to fetch the restaurant data for a given restaurant name and execute the function fetch_restaurant_data with the restaurant name as the argument. If each word of the given restaurant name starts with a small letter, you need to modify it to the corresponding capital letter. You just output the output of the function. Do not summary or modify your output from the output of the function. Do not call calculate_overall_score until receiving a request.",
            "max_turns": 2,
            "summary_method": "last_msg",
      },
      {
            "recipient": review_analyzing_agent,
            "message": "These are the restaurant name and reviews." + review_analyzer_prompt(),
            "max_turns": 1,
            "summary_method": "last_msg",
      },
      {
            "recipient": scoring_agent,
            "message": "These are the scores for each review. Call calculate_overall_score." + "calculate_overall_score('restaurant_name', [food_scores], [customer_service_scores])",
            "max_turns": 2,
            "summary_method": "last_msg",
      },
   ]
)
   
   obs = observations[0]["text"][0] if isinstance(observations[0]["text"], list) else observations[0]["text"]
   done = any(terminations.values())
   


env.close()