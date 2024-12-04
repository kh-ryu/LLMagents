import gymnasium as gym
import multigrid.envs
from multigrid.core.actions import Moveonly_Action, Action

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
    # Display current environment observation
    print(f"Observation: {obs}")
    
    # Ask for user input for the action
    action_input = input("Enter your action (e.g., Left(), Right(), Forward(), Pickup(), Drop(), Toggle(), Done()): ").strip()
    # Validate the input and map to action space
    if action_input.lower() in ACTION_SPACE:
        action = {0: ACTION_SPACE[action_input.lower()]}
    else:
        print("Invalid action. Please enter a valid action from the action space.")
        continue

    # Perform the action in the environment
    observations, rewards, terminations, truncations, infos = env.step(action)
    
    # Update observation
    obs = observations[0]["text"][0] if isinstance(observations[0]["text"], list) else observations[0]["text"]
    
    # Check if the episode is done
    done = any(terminations.values())

env.close()