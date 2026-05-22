import sys, os
import oai

# Per-million-token USD rates.
#
# gpt-5.5 / gpt-5.4 / gpt-5.4-nano are verified against OpenAI's pricing docs
# (developers.openai.com/api/docs/pricing, May 2026). gpt-5.5 / gpt-5.4 have a
# long-context tier: once a single request's prompt exceeds
# prompt_tier_threshold_tokens, the whole request bills at the *_above_threshold
# rates. Other entries are carried over from prior rates and have no verified
# cached rate, so cached input falls back to the uncached rate.
MC = {
    'gpt-5.5-2026-04-23': {
        'input_uncached': 5.00, 'input_cached': 0.50, 'output': 30.00,
        'prompt_tier_threshold_tokens': 272_000,
        'input_uncached_above_threshold': 10.00,
        'input_cached_above_threshold': 1.00,
        'output_above_threshold': 45.00,
    },
    'gpt-5.4-2026-03-05': {
        'input_uncached': 2.50, 'input_cached': 0.25, 'output': 15.00,
        'prompt_tier_threshold_tokens': 272_000,
        'input_uncached_above_threshold': 5.00,
        'input_cached_above_threshold': 0.50,
        'output_above_threshold': 22.50,
    },
    'gpt-5.4-nano': {'input_uncached': 0.20, 'input_cached': 0.02, 'output': 1.25},
    'gpt-5-nano-2025-08-07': {'input_uncached': 0.05, 'output': 0.40},
    'gpt-5.1-2025-11-13': {'input_uncached': 1.25, 'output': 10.00},
    'gpt-5-2025-08-07': {'input_uncached': 1.25, 'output': 10.00},
    'gpt-4.1-2025-04-14': {'input_uncached': 2.00, 'output': 8.00},
    'gpt-4o-2024-08-06': {'input_uncached': 2.50, 'output': 10.00},
    'gpt-4o-mini-2024-07-18': {'input_uncached': 0.15, 'output': 0.60},
    'gpt-4-0613': {'input_uncached': 30.00, 'output': 60.00},
    'gpt-4o-2024-05-13': {'input_uncached': 5.00, 'output': 15.00},
    'gpt-3.5-turbo-0613': {'input_uncached': 1.50, 'output': 2.00},
    'o1-2024-12-17': {'input_uncached': 15.00, 'output': 60.00},
    'o3-2025-04-16': {'input_uncached': 2.00, 'output': 8.00},
    'o1-mini-2024-09-12': {'input_uncached': 1.10, 'output': 4.40},
    'o4-mini-2025-04-16': {'input_uncached': 1.10, 'output': 4.40},
}

def get_model_cost(mn, it, ot, ct=0.):

    if it is None or ot is None:
        print("Missed a file")
        return 0.

    p = MC[mn]
    ct = ct or 0.

    # Long-context tier: when a single request's prompt (it = total prompt
    # tokens, cached included) exceeds the threshold, the whole request bills
    # at the above-threshold rates.
    threshold = p.get('prompt_tier_threshold_tokens')
    above = threshold is not None and it > threshold

    if above:
        r_unc = p.get('input_uncached_above_threshold', p['input_uncached'])
        r_cac = p.get('input_cached_above_threshold',
                      p.get('input_cached', p['input_uncached']))
        r_out = p.get('output_above_threshold', p['output'])
    else:
        r_unc = p['input_uncached']
        r_cac = p.get('input_cached', p['input_uncached'])
        r_out = p['output']

    cost = (it - ct) * r_unc + ct * r_cac + ot * r_out
    return cost / 1_000_000.

def get_cost_from_log_file(log_file):
    model_name = None
    in_tokens = None
    out_tokens = None
    cached_tokens = 0.
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
            elif k == 'cached_tokens':
                cached_tokens = float(v)
            elif k == 'duration':
                duration = float(v)

    return get_model_cost(model_name, in_tokens, out_tokens, cached_tokens), duration

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
