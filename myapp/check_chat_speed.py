import time
from concurrent.futures import ThreadPoolExecutor
from myapp import ai_chat


def probe(key):
    started = time.monotonic()
    try:
        cfg = ai_chat.MODELS[key]
        client = ai_chat._get_client(cfg.get('api_key_setting'))
        result = client.chat.completions.create(
            model=cfg['id'], messages=[{'role': 'user', 'content': 'Reply OK'}],
            max_tokens=32, timeout=15,
            extra_body={'chat_template_kwargs': {'enable_thinking': False}},
        )
        print(key, round(time.monotonic()-started, 1), result.choices[0].message.content, flush=True)
    except Exception as exc:
        print(key, type(exc).__name__, round(time.monotonic()-started, 1), flush=True)


def run():
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(probe, ['sol', 'quick']))
