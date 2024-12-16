from .llmagent import LLMAgent
from .action import *
from typing import List, Dict, Optional
from multigrid.core.actions import Moveonly_Action, Action


ACTION_SPACE = {
   "left()": Action.left,
   "right()": Action.right,
   "forward()": Action.forward,
   "pickup()": Action.pickup,
   "drop()": Action.drop,
   "toggle()": Action.toggle,
   "done()": Action.done
}

class MultiAgent:
    def __init__(self, env, agents, *args, **kwargs):
        self.agents = agents
        self.env = env

    def solve(self, obs: Dict[int, str], max_steps: int=30):
        self.env.reset()
        done = False
        for i in range(max_steps):
            actions = {}
            current_result = {}
            for agent in self.agents:
                current_result[agent.index] = {}

                agent_action = agent.response(obs[agent.index])
                assert agent_action is not None
                current_result[agent.index]["action"] = agent_action
                action = ACTION_SPACE[agent_action.params["name"]]
                actions[agent.index] = action
                observations, rewards, terminations, truncations, infos = self.env.step(actions)
            for agent_index, observation in observations.items():
                obs[agent_index] = observation["text"][0] if isinstance(observation["text"], list) else observation["text"]   
                current_result[agent_index]["obs"] = obs[agent_index]

            done = any([rewards[idx] for idx in rewards.keys()])
            if done:
                break
            self.chat(current_result)
            
        return done

    def chat(self, current_result, max_turns: int=3):
        obs = ["You are going to chat now. Please communicate with the other agent to share information and complete the task more effectively.\n"]
        print("Starting Chat Session...")
        for turn in range(max_turns):
            print(f"--- Turn {turn + 1} ---")
            # Each agent responds sequentially
            for i, agent in enumerate(self.agents):
                # Agent generates a response based on the latest observation
                action = agent.talk(obs[-1], current_action=current_result[agent.index]["action"], current_obs=current_result[agent.index]["obs"])
                messgae = action.params.get("message", "")
                if action is None or not isinstance(action, Chat):
                    print("Failed to parse action...")
                    return
                else:
                    obs.append(f"Agent {agent.index} says: {messgae}")

    
                        

            
    
    