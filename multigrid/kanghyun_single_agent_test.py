import gymnasium as gym
import multigrid.envs
import openai, os, re
from openai import OpenAI
from multigrid.core.actions import Moveonly_Action, Action
from prompt.utils import file_to_string, gpt_interaction, parse_action


openai.api_key = os.getenv('OPENAI_API_KEY')

system_prompt_folder = "kanghyun_prompt"
user_prompt_folder = "kanghyun_prompt"

system_prompt = file_to_string(f"./prompt/{system_prompt_folder}/single_system_prompt.txt")


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

obs = file_to_string(f"./prompt/{user_prompt_folder}/user_prompt_ball.txt")

observations, infos = env.reset()
text_obs = observations[0]["text"][0] if isinstance(observations[0]["text"], list) else observations[0]["text"]
done = False
is_ball_moved = False
is_door_open = False
history = [{"role": "user", "content": text_obs}]
history_max_len = 10

while not done:
   if is_ball_moved is False and is_door_open is False:
      user = file_to_string(f"./prompt/{user_prompt_folder}/user_prompt_ball.txt")
      prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]
   elif is_ball_moved is True and is_door_open is False:
      user = file_to_string(f"./prompt/{user_prompt_folder}/user_prompt_key.txt")
      prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]
   elif is_ball_moved is True and is_door_open is True:
      user = file_to_string(f"./prompt/{user_prompt_folder}/user_prompt_goal.txt")
      prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]

   messages = prompt + history
   print(f"Observation: {text_obs}")
   response = gpt_interaction(client, "gpt-4o", messages)
   history.append({"role": "assistant", "content": response})
   print(f"Assistant: {response}")
   action = parse_action(response, ACTION_SPACE)
   if action is None:
      obs = "Failed to parse your action."
      continue
   
   observations, rewards, terminations, truncations, infos = env.step(action)
   text_obs = observations[0]["text"][0] if isinstance(observations[0]["text"], list) else observations[0]["text"]
   history.append({"role": "user", "content": text_obs})
   if len(history) > history_max_len:
      history = history[-history_max_len:]

   is_ball_moved = env.ball_moved()
   is_door_open = env.door_open()

   done = any(terminations.values())
   
env.close()