

import gymnasium as gym
import multigrid.envs
import copy
import openai, os, re
from typing import List, Union
from .action import *
from openai import OpenAI
from multigrid.core.actions import Moveonly_Action, Action
from prompt.utils import file_to_string, gpt_interaction
from multigrid.core.agent import Agent, MissionSpace


CHAT_SYSTEM_PROMPT = f"""
You will communicate and share information with other agents. Through conversation, you will gather relevant details from them and collaboratively decide on the next course of action. Use the Chat Action to send and receive messages.
Action Space:
{Chat.action_description()}
You Could ONLY Use the Chat Action in your response.
"""

SUMMARIZE_PROMPT = """
After Chating with your teammate, please based on your responsibilities, historical exploration, and previous interactions with teammates, update your plans, beliefs, and understanding of the environment.
Your response should include:

Thought: A brief description of your thought process.
Update Information: The Update Information should be presented as a JSON object containing two key components:
	1. Belief: Describe your current understanding and perception of the environment after chat with your teammate. This should reflect how you interpret available data, past experiences, and team interactions.
	2. Plan: Outline your intended course of action. This should include specific strategies or steps you plan to take to achieve your goals while considering potential challenges and adjustments.

Response Format:
Thought: ...
Update Information: 
{
    "Belief": "Your current understanding and perception of the environment based on gathered information, past experiences, and team interactions.",
    "Plan": "Your proposed course of action and strategy to effectively accomplish the assigned tasks, considering current goals and potential challenges."
}


"""

class LLMAgent(Agent):
    def __init__(self, sys_prompt_path, mission_str, availiable_action: List[Union[AgentAction, str]], window_size:int=10, *args, **kwargs):    
        super().__init__(*args, **kwargs)
        self.mission_str = mission_str
        self.llm = OpenAI()
        self.availiable_action = availiable_action
        self.system_prompt = file_to_string(sys_prompt_path)
        self.chat_history = []
        self.history = []
        self.plan = ""
        self.belief = ""
        self.window_size = window_size
    
    def response(self, obs: str):
        # if self.chat_history:
        #   self.summarize_chat()
        action = None
        availiable_action_description = "\n".join(action.action_description() for action in self.availiable_action)
        system_prompt  = self.system_prompt.replace("===action===", availiable_action_description)
        system_prompt = system_prompt.replace("===mission===", self.mission_str)
        # system_prompt = system_prompt.replace("===belief===", self.belief)
        # system_prompt = system_prompt.replace("===plan===", self.plan)
        turn = 0
        chat_msg = "Chat History:\n" if self.chat_history else ""
        for msg in self.chat_history:
            if msg['role'] == "user":
                chat_msg += f"Turn {turn}:\n"
                chat_msg += f"{msg['role']}: {msg['content']}\n"
            else:
                parsed_msg = self.parse_action(msg['content'], availiable_actions=[Chat])
                chat_msg += f"You: {msg['content']}\n" if not isinstance(parsed_msg, Chat) else f"You: {parsed_msg.params.get('message', '')}"
                
            if msg['role'] == "assistant":
                turn += 1
        self.chat_history = []    
        chat_obs = chat_msg + "\n" + obs
        self.history.append({"role": "user", "content": chat_obs})
        active_history = [{"role": "system", "content": system_prompt}]+ copy.deepcopy(self.history)
       
        print(f"Agent {self.index} Color: {self.color}")
        print(f"Agent {self.index} Observation: {chat_obs}")
    
        for i in range(3):
            response = gpt_interaction(self.llm, "gpt-4o",  active_history)
            print(f"Agent {self.index} Response: {response}")
            
            try:
                action = self.parse_action(response)
                if action:
                    break
                else:
                    msg = "Failed to parse your Action. You must choose One Action from your action space. Please try again."
                    active_history.append({"role": "assistant", "content": response})
                    active_history.append({"role": "user", "content": msg})
                    continue
            except Exception as e:
                msg = "Failed to parse your Action. You must choose One Action from your action space, and None Action is not allowed. Please try again."
                active_history.append({"role": "assistant", "content": response})
                active_history.append({"role": "user", "content": msg})
                continue
                
       
        self.history.append({"role": "assistant", "content": response})
        self.history[:] = self.history[-self.window_size:]
        
        return action
    

    def parse_action(self, output: Optional[Union[List[str], str]], **kwargs) -> AgentAction:
        availiable_actions = kwargs.get('availiable_actions', [])
        availiable_actions = availiable_actions if availiable_actions else self.availiable_action
        if output is None or len(output) == 0:
            pass
        action_string = ""
        patterns = [ r'["\']?Action["\']?:? (.*?)Action', 
            r'["\']?Action["\']?:? (.*?)Observation',r'["\']?Action["\']?:? (.*?)Thought', 
            r'["\']?Action["\']?:? (.*?)$', r'^(.*?)Observation']
        for p in patterns:
            match = re.search(p, output, flags=re.DOTALL)
            if match:
                action_string = match.group(1).strip()
                break
        if action_string == "":
            action_string = output.strip()
      
        output_action = None
        for action_cls in availiable_actions:
            action = action_cls.parse_action_from_text(action_string)
            if action is not None:
                output_action = action
                break
        if output_action is None:
            action_string = action_string.replace("\_", "_").replace("'''","```")
            for action_cls in availiable_actions:
                action = action_cls.parse_action_from_text(action_string)
                if action is not None:
                    output_action = action
                    break
        
        return output_action
    
    def talk(self, msg, current_action, current_obs):
          
        availiable_action = copy.deepcopy(self.availiable_action)
        self.availiable_action = [Chat]

        combined_history = (
            "### Summary of Your Progress ###\n\n"
            f"Your Mission: {self.mission_str}\n\n"
            f"Your Exploration History:\n" +
            "\n".join(
                f"{message['role']}: {message['content']}" if message["role"] != "assistant" 
                else f"You: {message['content']}" 
                for message in self.history
            ) +
            f"\n\nYour Last Action: {current_action.action_to_text()}\n"
            f"Your Current Observation: {current_obs}\n"
        )
        
        self.chat_history.append({"role": "user", "content": msg})
        messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT + "\n" + combined_history}] + self.chat_history 

        for i in range(3):
            print(f"Agent {self.index} Recieve: {msg}")
            response = gpt_interaction(self.llm, "gpt-4o", messages)
            print(f"Agent {self.index} Response: {response}")
            try:
                action = self.parse_action(response)
                if action:
                    break
                else:
                    msg = "Failed to extract an Action. You can now only use the Chat Action to communicate with other agents and cannot perform any other actions. Please try again."
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": msg})
                    continue
            except Exception as e:
                msg = f"Error parsing action: {e}"
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": msg})
                continue
        self.availiable_action = availiable_action
        # Update the selected history with the assistant's response
        self.chat_history.append({"role": "assistant", "content": response})
        return action

        
    def summarize_chat(self): 
        turn = 1
        chat_msg = ""
        for msg in self.chat_history:
            if msg['role'] == "user":
                chat_msg += f"Turn {turn}:\n"
                chat_msg += f"{msg['role']}: {msg['content']}\n"
            else:
                parsed_msg = self.parse_action(msg['content'], availiable_actions=[Chat])
                chat_msg += f"You: {msg['content']}\n" if not isinstance(parsed_msg, Chat) else f"You: {parsed_msg.params.get('message', '')}"
                
            if msg['role'] == "assistant":
                turn += 1
        
        prompt = (
            "Your Mission and Goal: {self.mission}"
            f"Your Exploration History:\n" +
            "\n".join(
                f"{message['role']}: {message['content']}" if message["role"] != "assistant" 
                else f"You: {message['content']}" 
                for message in self.history
            ) +
            f"Your Chat History with teammate:\n {chat_msg}"
            f"Your Current Belief: {self.belief}"
            f"Your Current Plan: {self.plan}"
        )

        
        messages = [{"role": "system", "content": SUMMARIZE_PROMPT}, {"role": "user", "content": prompt}]
        
        for attempt in range(3):
            print(f"Agent {self.index} Summarize (Attempt {attempt + 1}):")
            response = gpt_interaction(self.llm, "gpt-4o", messages)
            print(f"Agent {self.index} Response: {response}")
            
            pattern = r"\{.*?\}"
            matches = re.findall(pattern, response, re.DOTALL)
 
            for match in matches:
                dict_str = match.replace("\n", "").replace("\t", "")
               
                try:
                    result = json.loads(dict_str)
                    belief = result.get("Belief", "")
                    plan = result.get("Plan", "")
                    
                    
                    self.belief = belief if belief else ""
                    self.plan = plan if plan else ""
                    
                    self.chat_history = []  
                    return 
                
                except json.JSONDecodeError:
                    continue
        
        self.chat_history = []
        return None
            
            

        
        
        
       
    
    
        
        
        
        