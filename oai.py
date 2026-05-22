import time
import os

from openai import OpenAI
import copy


import datetime

COST_LOG_DIR = PATH_TO_COST_LOG
CALL_LOG_DIR = PATH_TO_CALL_LOG
LOG_DIR = COST_LOG_DIR

API_KEY = None

os.makedirs(COST_LOG_DIR, exist_ok=True)
os.makedirs(CALL_LOG_DIR, exist_ok=True)

MM = {
    'gpt5.5': 'gpt-5.5',
    'gpt5.4nano': 'gpt-5.4-nano',
    'gpt5.4': 'gpt-5.4',    
    'gpt5.1': 'gpt-5.1',
    'gpt5': 'gpt-5', 
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


assert API_KEY is not None, "TODO SET API_KEY"

client = OpenAI(
    api_key = API_KEY,
)

    
def get_session_name():
    now = datetime.datetime.now()
    tm = str(now).replace(':','_').replace('.','_').replace(' ','_').replace('-','_')
    return tm

def parse_response(R, duration):
    response_id = getattr(R, "id", None)
    created = getattr(R, "created", None)
    model = R.model
    text = R.choices[0].message.content
    finish = R.choices[0].finish_reason
    service_tier = getattr(R, "service_tier", None)
    system_fingerprint = getattr(R, "system_fingerprint", None)

    input_tokens = getattr(R.usage, "prompt_tokens", None)
    output_tokens = getattr(R.usage, "completion_tokens", None)
    total_tokens = getattr(R.usage, "total_tokens", None)

    cached_tokens = None
    if hasattr(R.usage, "prompt_tokens_details") and R.usage.prompt_tokens_details:
        cached_tokens = getattr(R.usage.prompt_tokens_details, "cached_tokens", None)

    reasoning_tokens = None
    if hasattr(R.usage, "completion_tokens_details") and R.usage.completion_tokens_details:
        reasoning_tokens = getattr(R.usage.completion_tokens_details, "reasoning_tokens", None)

    return {
        "time": datetime.datetime.now(),
        "response_id": response_id,
        "created": created,
        "duration": duration,
        "model": model,
        "service_tier": service_tier,
        "system_fingerprint": system_fingerprint,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
        "finish": finish,
        "text": text,
        "reasoning_tokens": reasoning_tokens,
    }


def sanitize_for_log(value):
    value = copy.deepcopy(value)

    if isinstance(value, dict):
        sanitized = {}
        for k, v in value.items():
            if k == "url" and isinstance(v, str) and v.startswith("data:image"):
                sanitized[k] = "[image omitted from log]"
            else:
                sanitized[k] = sanitize_for_log(v)
        return sanitized

    if isinstance(value, list):
        return [sanitize_for_log(v) for v in value]

    return value


def get_cost_info(info):
    cost_keys = [
        "status",
        "time",
        "model_alias",
        "model_requested",
        "duration",
        "model",
        "service_tier",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_tokens",
        "finish",
        "reasoning_tokens",
        "call_log",
        "error_type",
        "error",
    ]
    return {k: info.get(k) for k in cost_keys}


def log_info(f, info):
    for k,v in info.items():
        f.write(f'{k}:{v}\n')

def log_error(f, e, mn, ms):
    f.write(f"Error: {e}\n")
    f.write(f"model: {mn}\n")
    f.write(f"messages: {sanitize_for_log(ms)}\n")
    
def query_model(
    model_name,
    messages,
    **kwargs
):

    sn = get_session_name()
    cost_fn = f'{COST_LOG_DIR}/{sn}.txt'
    call_fn = f'{CALL_LOG_DIR}/{sn}.txt'
    model = None

    try:
        model = MM[model_name]
        t = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        duration = time.time() - t
        info = parse_response(response, duration)
        info['status'] = 'success'
        info['model_alias'] = model_name
        info['model_requested'] = model
        info['call_log'] = call_fn
        call_info = dict(info)
        call_info['input'] = sanitize_for_log(messages)
        call_info['request_kwargs'] = sanitize_for_log(kwargs)

        with open(cost_fn, 'w') as f:
            log_info(f, get_cost_info(info))

        with open(call_fn, 'w') as f:
            log_info(f, call_info)

        return info['text'], cost_fn

    except Exception as e:
        error_info = {
            "status": "error",
            "time": datetime.datetime.now(),
            "model_alias": model_name,
            "model_requested": model,
            "error_type": type(e).__name__,
            "error": str(e),
            "call_log": call_fn,
        }
        call_error_info = dict(error_info)
        call_error_info['input'] = sanitize_for_log(messages)
        call_error_info['request_kwargs'] = sanitize_for_log(kwargs)

        with open(cost_fn, 'w') as f:
            log_info(f, get_cost_info(error_info))

        with open(call_fn, 'w') as f:
            log_info(f, call_error_info)

        return None, None

def base_query(
    model,
    user_message,
    system_message = None,
    log_file=None,
    image_data=None,
    **kwargs    
):

    messages = []

    if system_message is not None:
        messages.append(
            {'role': 'developer', 'content': system_message},            
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
        messages,
        **kwargs
    )

    if log_file is not None and lfn is not None:
        with open(log_file, 'a') as f:
            f.write(f'{lfn}\n')
    
    return output


def conv_query(
    model,
    messages,
    log_file=None,
    **kwargs
):
    
    output, lfn = query_model(
        model,
        messages,
        **kwargs        
    )

    if log_file is not None and lfn is not None:
        with open(log_file, 'a') as f:
            f.write(f'{lfn}\n')
    
    return output

def test():
    import sys
    
    model_name = sys.argv[1]
    message = ' '.join(sys.argv[2:])

    base_query(model_name, message, reasoning_effort='high')
    
if __name__ == '__main__':
    test()
