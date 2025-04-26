import time
import os
import openai
import datetime

LOG_DIR = PATH_TO_LOG

os.system(f'mkdir {LOG_DIR} > /dev/null 2>&1')

MM = {
    'gpt3.5': 'gpt-3.5-turbo',
    'gpt4': 'gpt-4',
    'gpt4o': 'gpt-4o',
    'gpt4p1': 'gpt-4.1',
    'gpt4omini': 'gpt-4o-mini',
    'o1mini': 'o1-mini',
    'o4mini': 'o4-mini',
    'o1': 'o1',
    'o3': 'o3',
    'may_gpt4o': 'gpt-4o-2024-05-13'
}

openai.organization = ORG_KEY
openai.api_key = API_KEY

def get_session_name():
    now = datetime.datetime.now()
    tm = str(now).replace(':','_').replace('.','_').replace(' ','_').replace('-','_')
    return tm

def parse_response(R, duration):
    model = R['model']
    text = R['choices'][0]['message']['content']
    finish = R['choices'][0]['finish_reason']

    input_tokens = R['usage']['prompt_tokens']
    output_tokens = R['usage']['completion_tokens']
    total_tokens = R['usage']['total_tokens']

    return {
        'time': datetime.datetime.now(),
        'duration': duration,
        'model': model,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': total_tokens,
        'finish': finish,                
        'text': text,
    }

def log_info(f, info):
    for k,v in info.items():
        f.write(f'{k}:{v}\n')

def log_error(f, e, mn, ms):
    f.write(f"Error: {e}\n")
    f.write(f"model: {mn}\n")
    f.write(f"messages: {ms}\n")
    
def query_model(
    model_name,
    messages,
):

    sn = get_session_name()
    fn = f'{LOG_DIR}/{sn}.txt'
    f = open(f'{LOG_DIR}/{sn}.txt', 'w')

    try:
        model = MM[model_name]
        t = time.time()        
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages
        )
        duration = time.time() - t
        info = parse_response(response, duration)
        info['input'] = messages
        log_info(f, info)
        return info['text'], fn

    except Exception as e:
        log_error(f, e, model_name, messages)                  
        return None, None

def base_query(
    model,
    user_message,
    system_message = None,
    log_file=None,
    image_data=None    
):

    messages = []

    if system_message is not None:
        messages.append(
            {'role': 'system', 'content': system_message},            
        )

    if image_data is None:
        messages.append(
            {'role': 'user', 'content': user_message}
        )
    else:
        messages.append(
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': user_message
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ],
            }
        )
    
    output, lfn = query_model(
        model,
        messages
    )

    if log_file is not None and lfn is not None:
        with open(log_file, 'a') as f:
            f.write(f'{lfn}\n')
    
    return output


def conv_query(
    model,
    messages,
    log_file=None,
):
    
    output, lfn = query_model(
        model,
        messages
    )

    if log_file is not None and lfn is not None:
        with open(log_file, 'a') as f:
            f.write(f'{lfn}\n')
    
    return output


import base64
# Function to encode the image
def oai_enc_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
