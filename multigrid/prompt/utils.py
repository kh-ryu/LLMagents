import openai
import os
import re

def file_to_string(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()
    
def gpt_interaction(client, gpt_model, messages):
    trial = 0
    completion = None
    
    while completion is None and trial < 5:
        completion = client.chat.completions.create(
            model=gpt_model,
            messages=messages,
            max_tokens=1000,
            temperature=0.5
        )
        trial += 1

    return completion.choices[0].message.content

def save_string_to_file(save_path, string_file):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as file:
        file.write(string_file)

def parse_action(response, action_space):
    pattern = r"Action:\s*([A-Za-z]+\(.*?\))"
    match = re.search(pattern, response)
    if match:
        action = match.group(1)
        action = {0: action_space[action.lower().strip()]}
    else:
        print("Failed to parse your action.")
        action = None
    return action