"""Build the real-prompt workloads for the larger-model sweep.

Runs on the GPU host (it needs the model's tokenizer from Hugging Face).

Sources (both public):
  ShareGPT V3: first user turn of each conversation, used as the user prompt.
  awesome-chatgpt-prompts: community-written role prompts, used as shared
  system prompts for the prefix-caching workload.

Every prompt is written with the chat template already applied (thinking off),
so `vllm bench serve --dataset-name custom --skip-chat-template` sends exactly
the text whose length we measured here.

Outputs in --out:
  len<L>.jsonl   prompts whose templated length is within +/-TOL of L tokens
  natural.jsonl  every usable prompt up to --max-len tokens (real length mix)
  prefix.jsonl   one of --num-prefixes role prompts as the system message,
                 plus a user prompt of at most 1,024 tokens
  stats.json     counts and token-length percentiles per file, source hashes
"""
import argparse, csv, hashlib, json, os, random, statistics


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def first_user_turns(sharegpt_path):
    data = json.load(open(sharegpt_path, encoding='utf-8'))
    for entry in data:
        turns = entry.get('conversations') or []
        if turns and turns[0].get('from') == 'human':
            text = (turns[0].get('value') or '').strip()
            if text:
                yield text


def templated(tok, messages, generation_prompt=True):
    return tok.apply_chat_template(messages, add_generation_prompt=generation_prompt,
                                   tokenize=False, enable_thinking=False)


def n_tokens(tok, text):
    # The benchmark client counts prompt tokens the same way.
    return len(tok(text).input_ids)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def describe(rows):
    lens = [r['tokens'] for r in rows]
    if not lens:
        return {'count': 0}
    return {'count': len(lens), 'mean': round(statistics.mean(lens), 1),
            'median': statistics.median(lens), 'p10': pct(lens, 0.10),
            'p90': pct(lens, 0.90), 'min': min(lens), 'max': max(lens)}


def write_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps({'prompt': r['prompt']}, ensure_ascii=False) + '\n')


def main(argv=None, tokenizer=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--sharegpt', required=True)
    ap.add_argument('--role-prompts', required=True, help='awesome-chatgpt-prompts CSV')
    ap.add_argument('--out', required=True)
    ap.add_argument('--lengths', default='512,1024,2048,4096')
    ap.add_argument('--tol', type=float, default=0.2)
    ap.add_argument('--max-len', type=int, default=4096)
    ap.add_argument('--num-prefixes', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(argv)

    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(a.tokenizer)
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)

    seen, users = set(), []
    n_raw = n_dup = 0
    for text in first_user_turns(a.sharegpt):
        n_raw += 1
        if text in seen:
            n_dup += 1
            continue
        seen.add(text)
        p = templated(tokenizer, [{'role': 'user', 'content': text}])
        users.append({'text': text, 'prompt': p, 'tokens': n_tokens(tokenizer, p)})
    rng.shuffle(users)

    stats = {'sources': {'sharegpt_sha256': sha256(a.sharegpt),
                         'role_prompts_sha256': sha256(a.role_prompts)},
             'sharegpt_first_user_turns': n_raw, 'exact_duplicates_dropped': n_dup,
             'tolerance': a.tol, 'files': {}}

    for L in [int(x) for x in a.lengths.split(',')]:
        lo, hi = L * (1 - a.tol), L * (1 + a.tol)
        rows = [u for u in users if lo <= u['tokens'] <= hi]
        write_jsonl(os.path.join(a.out, f'len{L}.jsonl'), rows)
        stats['files'][f'len{L}.jsonl'] = describe(rows)

    natural = [u for u in users if u['tokens'] <= a.max_len]
    write_jsonl(os.path.join(a.out, 'natural.jsonl'), natural)
    stats['files']['natural.jsonl'] = describe(natural)

    roles = []
    with open(a.role_prompts, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if 'prompt' not in (reader.fieldnames or []):
            raise SystemExit(f'{a.role_prompts} has no "prompt" column: {reader.fieldnames}')
        for row in reader:
            text = (row.get('prompt') or '').strip()
            if text:
                roles.append(text)
    rng.shuffle(roles)
    roles = roles[:a.num_prefixes]
    if len(roles) < a.num_prefixes:
        raise SystemExit(f'only {len(roles)} role prompts, need {a.num_prefixes}')
    short = [u for u in users if u['tokens'] <= 1024]
    prefix_rows, shares = [], []
    for i, u in enumerate(short):
        system = roles[i % len(roles)]
        p = templated(tokenizer, [{'role': 'system', 'content': system},
                                  {'role': 'user', 'content': u['text']}])
        sys_only = templated(tokenizer, [{'role': 'system', 'content': system}],
                             generation_prompt=False)
        total = n_tokens(tokenizer, p)
        prefix_rows.append({'prompt': p, 'tokens': total})
        # Shared part: the system block, which every request with this role starts with.
        shares.append(min(1.0, n_tokens(tokenizer, sys_only) / total))
    write_jsonl(os.path.join(a.out, 'prefix.jsonl'), prefix_rows)
    stats['files']['prefix.jsonl'] = describe(prefix_rows)
    stats['files']['prefix.jsonl']['num_prefixes'] = len(roles)
    if shares:
        stats['files']['prefix.jsonl']['approx_shared_fraction_median'] = round(statistics.median(shares), 3)

    json.dump(stats, open(os.path.join(a.out, 'stats.json'), 'w'), indent=2)
    print(json.dumps(stats['files'], indent=2))


if __name__ == '__main__':
    main()
