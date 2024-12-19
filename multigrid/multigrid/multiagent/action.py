
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional
import re
import json

from multigrid.core.actions import Moveonly_Action, Action

@dataclass
class AgentAction(ABC):
    params: Optional[Dict] = field(
        metadata={"help": 'argument of action'}
    )
    
    @classmethod
    @abstractmethod
    def parse_action_from_text(cls, text):
        pass
    
    @abstractmethod
    def action_to_text(self):
        pass
    
    @classmethod
    @abstractmethod
    def action_description(cls):
        pass
    
class Forward(AgentAction):
    params = {"name": "forward()"}

    @classmethod
    def parse_action_from_text(cls, text):
        text = text.lower()
        match = re.match(r"forward\((.*?)\)", text.strip(), re.DOTALL)
        if match:
           forward = match.group(1).strip()
           return cls(params={"name": "forward()"})
        else:
            return None
        
    def action_to_text(self):
        return "Forward()"
    
    @classmethod
    def action_description(cls):
        return """
## Forward Action
* Signature: Forward()
* Description: Moves the agent one cell forward.
* Example:
Action: Forward()
"""
        

class Left(AgentAction):
    params = {"name": "left()"}

    @classmethod
    def parse_action_from_text(cls, text):
        text = text.lower()
        match = re.match(r"left\((.*?)\)", text.strip(), re.DOTALL)
        if match:
           action = match.group(1).strip()
           return cls(params={"name": "left()"})
        else:
            return None
        
    def action_to_text(self):
        return "Left()"

    @classmethod
    def action_description(cls):
        return """
## Left Action
* Signature: Left()
* Description: Turns the agent 90 degrees to the left.
* Example: 
Action: Left()
"""

class Right(AgentAction):
    params = {"name": "right()"}

    @classmethod
    def parse_action_from_text(cls, text):
        text = text.lower()
        match = re.match(r"right\((.*?)\)", text.strip(), re.DOTALL)
        if match:
           action = match.group(1).strip()
           return cls(params={"name": "right()"})
        else:
            return None
        
    def action_to_text(self):
        return "Right()"
    
    @classmethod
    def action_description(cls):
        return """
## Right Action
* Signature: Right()
* Description: Turns the agent 90 degrees to the right.
* Example:
Action: Right()
"""

class Pickup(AgentAction):
    params = {"name": "pickup()"}

    @classmethod
    def parse_action_from_text(cls, text):
        text = text.lower()
        match = re.match(r"pickup\((.*?)\)", text.strip(), re.DOTALL)
        if match:
           action = match.group(1).strip()
           return cls(params={"name": "pickup()"})
        else:
            return None
        
    def action_to_text(self):
        return "Pickup()"

    @classmethod
    def action_description(cls):
        return """
## Pickup Action
* Signature: Pickup()
* Description: Attempts to pick up an object directly in front of the agent. Note that only objects that are eligible for pickup can be picked up.
* Example:
Action: Pickup() 
"""

class Toggle(AgentAction):
    params = {"name": "toggle()"}

    @classmethod
    def parse_action_from_text(cls, text):
        text = text.lower()
        match = re.match(r"toggle\((.*?)\)", text.strip(), re.DOTALL)
        if match:
           action = match.group(1).strip()
           return cls(params={"name": "toggle()"})
        else:
            return None
        
    def action_to_text(self):
        return "Toggle()"
    
    @classmethod
    def action_description(cls):
        return """
## Toggle Action
* Signature: Toggle()
* Description: Attempt to open a door in front of the agent. The action will only succeed if the agent is carrying the appropriate key required to unlock the door.
* Example:
Action: Toggle()
"""

class Drop(AgentAction):
    params = {"name": "drop()"}

    @classmethod
    def parse_action_from_text(cls, text):
        text = text.lower()
        match = re.match(r"drop\((.*?)\)", text.strip(), re.DOTALL)
        if match:
           action = match.group(1).strip()
           return cls(params={"name": "drop()"})
        else:
            return None
        
    def action_to_text(self):
        return "Drop()"
    
    @classmethod
    def action_description(cls):
        return """
## Drop Action
* Signature: Drop()
* Description: Attempts to drop the object the agent is carrying.
* Example:
Action: Drop()
"""

     
class Done(AgentAction):
    params = {"name": "done()"}

    @classmethod
    def parse_action_from_text(cls, text):
        text = text.lower()
        match = re.match(r"done\((.*?)\)", text.strip(), re.DOTALL)
        if match:
           action = match.group(1).strip()
           return cls(params={"name": "done()"})
        else:
            return None
        

    def action_to_text(self):
        return "Done()"   
    
    @classmethod
    def action_description(cls):
        return """
## Done Action
* Signature: Done()
* Description: Indicates that the agent has completed the whole task.
* Example:
Action: Done()
"""
    
class Chat(AgentAction):
    params = {"name": "chat", "message": ""}
    
    @classmethod
    def parse_action_from_text(cls, text):
        text = text.lower()
        pattern = re.compile(r"(\{.*?\})", re.DOTALL)
        match = pattern.search(text)
        if match:
            dict_str = match.group(1)
            dict_str = dict_str.replace("\n", "").replace("\t", "")
            try:
                result = json.loads(dict_str)
                return cls(params={"message": result.get("message", ""), "name": "chat"})
            except json.JSONDecodeError:
                
                return None
        return None
        
    def action_to_text(self):
        return f'{{"message": "{self.params["message"]}"\n}}'


    @classmethod
    def action_description(cls):
        return """
## Chat Action
* Signature: {"messgae": the message you want to send}
* Description: This action allows the agent to send a text-based message. The message field should contain the content of the message the agent intends to communicate. If the message field is left empty, no meaningful communication will occur.
* Example:
- Example 1:
Action: {"message": "I’m currently planning to find the key, but you’re standing in my way."}
- Example 2:
Action: {"message": "I’ve got the key now and am ready to open the door. Could you help move the ball away from the door?"}
"""
