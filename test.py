import gymnasium as gym
import multigrid.multigrid.envs

env = gym.make('MultiGrid-BlockedUnlockPickup-v0', agents=2, render_mode='human')
env = env.unwrapped

observations, infos = env.reset()
done = False
while not done:
   # this is where you would insert your policy / policies
   actions = {agent.index: agent.action_space.sample() for agent in env.agents}
   observations, rewards, terminations, truncations, infos = env.step(actions)
   done = any(terminations.values())

env.close()