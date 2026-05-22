import sys, os
import oai

MC = {
    'gpt-5.5-2026-04-23': {'i': 0.005, 'o': 0.03},
    'gpt-5.4-2026-03-05': {'i': 0.0025, 'o': 0.015},
    'gpt-5.4-nano': {'i': 0.0002, 'o': 0.00125},
    'gpt-5-nano-2025-08-07': {'i': 0.00005, 'o': 0.0004},
    'gpt-5.1-2025-11-13': {'i': 0.00125, 'o': 0.01},
    'gpt-5-2025-08-07': {'i': 0.00125, 'o': 0.01},
    'gpt-4.1-2025-04-14': {'i': 0.002, 'o': 0.008},
    'gpt-4o-2024-08-06': {'i': 0.0025, 'o': 0.01},
    'gpt-4o-mini-2024-07-18': {'i': 0.00015, 'o': 0.0006},
    'gpt-4-0613': {'i': 0.03, 'o': 0.06},
    'gpt-4o-2024-05-13' : {'i': 0.005, 'o':0.015},
    'gpt-3.5-turbo-0613': {'i': 0.0015, 'o': 0.002},    
    'o1-2024-12-17': {'i': .015, 'o': 0.06},
    'o3-2025-04-16': {'i': .002, 'o': 0.008},
    'o1-mini-2024-09-12': {'i': .0011, 'o': 0.0044},
    'o4-mini-2025-04-16': {'i': .0011, 'o': 0.0044},
    
}

def get_model_cost(mn, it, ot):

    cost = 0.

    if it is None or ot is None:
        print("Missed a file")
        return 0.
        
    
    cost += MC[mn]['i'] * it / 1000.
    cost += MC[mn]['o'] * ot / 1000.
    
    return cost

def get_cost_from_log_file(log_file):
    model_name = None
    in_tokens = None
    out_tokens = None
    duration = 0.
    
    with open(log_file) as f:
        for line in f:
            L = line.split(':', 1)
            if len(L) != 2:
                continue
            k,v = L[0].strip(),L[1].strip()

            if v.lower() == 'none':
                continue
            
            if k == 'model':
                model_name = v
            elif k == 'input_tokens':
                in_tokens = float(v)
            elif k == 'output_tokens':
                out_tokens = float(v)
            elif k == 'duration':
                duration = float(v)
                
    return get_model_cost(model_name, in_tokens, out_tokens), duration

def get_cost_from_exp_log_file(exp_log_file):
    log_files = []
    with open(exp_log_file) as f:
        for line in f:
            log_files.append(line.strip())

    print(f"Found {len(log_files)} log files")

    total_cost = 0.
    total_dur = 0.
    
    for lf in log_files:
        tc, td = get_cost_from_log_file(lf)
        total_cost += tc
        total_dur += td
        
    print(f"Total cost: {total_cost} ({round(total_dur, 2)})")

def get_cost_info_from_exp_log_file(exp_log_file):
    log_files = []
    with open(exp_log_file) as f:
        for line in f:
            log_files.append(line.strip())

    total_cost = 0.
    total_dur = 0.
    
    for lf in log_files:
        tc, td = get_cost_from_log_file(lf)
        total_cost += tc
        total_dur += td
        
    return total_cost, total_dur

    
def get_day_log_files(day):
    log_dirs = [
        getattr(oai, 'COST_LOG_DIR', None),
    ]
    log_files = []

    for log_dir in log_dirs:
        if log_dir is None or not os.path.isdir(log_dir):
            continue

        for lf in os.listdir(log_dir):
            nm = lf.split('/')[-1]

            if nm[:len(day)] == day:
                log_files.append(f'{log_dir}/{lf}')

    return log_files

def get_total_cost_from_day(day):
    day_logs = get_day_log_files(day)

    print(f"Found {len(day_logs)} number of log files")
    total_cost = 0.
    total_dur = 0.
    
    for lf in day_logs:
        tc, td = get_cost_from_log_file(lf)

        total_cost += tc
        total_dur += td
        
    print(f"Total day cost for {day}: {total_cost}")
    
if __name__ == '__main__':
    mode = sys.argv[1]
    
    if mode == 'exp':
        get_cost_from_exp_log_file(sys.argv[2])

    if mode == 'day':
        get_total_cost_from_day(sys.argv[2])
