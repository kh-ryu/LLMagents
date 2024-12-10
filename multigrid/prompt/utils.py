import openai
import os

def file_to_string(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()
    
def gpt_interaction(client, gpt_model, system_string, user_string):
    trial = 0
    completion = None
    
    while completion is None and trial < 5:
        completion = client.chat.completions.create(
            model=gpt_model,
            messages=[
            {"role": "system", "content": system_string},
            {"role": "user", "content": user_string}
            ]
        )
        trial += 1

    return completion.choices[0].message.content

def save_string_to_file(save_path, string_file):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as file:
        file.write(string_file)