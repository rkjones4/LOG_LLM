import sys
import time
import asyncio

import oai


DEFAULT_MODEL = "gpt5.4"
MODES = ("chat", "responses", "async")

# Experiment log file the test appends to, so a run can be cost-checked with:
#   python3 cost.py exp logs/test_exp_log.txt
EXP_LOG_FILE = "logs/test_exp_log.txt"


def usage():
    print('Usage: python3 test.py [mode] [model] "prompt"')
    print(f'  mode:  {" | ".join(MODES)}  (default: chat)')
    print(f'  model: an alias from oai.MM  (default: {DEFAULT_MODEL})')
    print()
    print('  chat       -> oai.base_query             (Chat Completions API)')
    print('  responses  -> oai.responses_query        (Responses API, sync)')
    print('  async      -> make_async_client + responses.create')
    print('                + log_responses_call       (the path Articraft uses)')
    print()
    print('Examples:')
    print('  python3 test.py "hello"')
    print('  python3 test.py responses "hello"')
    print('  python3 test.py async gpt5.4 "hello"')
    sys.exit(1)


def parse_args(argv):
    """Return (mode, model, prompt).

    The first argument is an optional mode keyword; if it is not a known mode
    the original `[model] "prompt"` form is assumed (so old invocations of this
    script still work and default to chat mode).
    """
    args = list(argv)
    mode = "chat"
    if args and args[0] in MODES:
        mode = args.pop(0)
    if not args:
        usage()
    if len(args) == 1:
        model = DEFAULT_MODEL
        prompt = args[0]
    else:
        model = args[0]
        prompt = " ".join(args[1:])
    return mode, model, prompt


def run_chat(model, prompt):
    """Chat Completions path (oai.base_query)."""
    return oai.base_query(model, prompt, log_file=EXP_LOG_FILE)


def run_responses(model, prompt):
    """New sync Responses API path (oai.responses_query)."""
    response = oai.responses_query(model, log_file=EXP_LOG_FILE, input=prompt)
    return getattr(response, "output_text", None)


def run_async(model, prompt):
    """The async path Articraft uses: build an AsyncOpenAI client via
    make_async_client, make the call directly, then log it with
    log_responses_call."""

    async def _run():
        aclient = oai.make_async_client()
        try:
            t = time.time()
            response = await aclient.responses.create(
                model=oai.MM.get(model, model),
                input=prompt,
            )
            oai.log_responses_call(
                response,
                time.time() - t,
                model_alias=model,
                request_kwargs={"input": prompt},
                log_file=EXP_LOG_FILE,
            )
            return getattr(response, "output_text", None)
        finally:
            await aclient.close()

    return asyncio.run(_run())


def main():
    if len(sys.argv) < 2:
        usage()

    mode, model, prompt = parse_args(sys.argv[1:])
    print(f"[mode={mode} model={model}]")

    if mode == "chat":
        response = run_chat(model, prompt)
    elif mode == "responses":
        response = run_responses(model, prompt)
    elif mode == "async":
        response = run_async(model, prompt)
    else:  # unreachable: parse_args restricts mode to MODES
        usage()

    if response is None:
        print("No response. Check the generated cost/call logs for the error.")
        sys.exit(1)

    print(response)
    print()
    print(f"Logged to {EXP_LOG_FILE} -- check cost with:")
    print(f"  python3 cost.py exp {EXP_LOG_FILE}")


if __name__ == "__main__":
    main()
