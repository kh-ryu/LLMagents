import gymnasium as gym
import multigrid.envs
import openai, os, re
import logging, yaml
from openai import OpenAI
from multigrid.core.actions import Moveonly_Action, Action
from prompt.utils import file_to_string
from multigrid.core.agent import Agent, MissionSpace
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Any
from config import AgentConfigModel


class MulitAgent:
    def __init__(self, config_file: str):
    
        self._load_cfg(config_file)
            
    def _load_cfg(self, cfg_file):
        try:
            with open(cfg_file, 'r') as f:
                config_data = yaml.safe_load(f)
            self.config = AgentConfigModel(**config_data)
            logging.error("Configuration loaded successfully:", self.config)
        except FileNotFoundError:
            logging.error("Configuration file not found!")
        except ValidationError as e:
            logging.error("Configuration validation failed!")
            logging.error(e)

