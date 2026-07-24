"""Shared pieces of the on-device continual-triage demo (see the accompanying blog post).

The synthetic drifting inbox, the (category, regime) policy, prompt rendering, and
the LFM2.5-230M model helpers used by both entry points:

  run_baselines.py   ZS / ICL / RAG arms (+ both k sweeps) -> outputs/baselines.json
  run_sft.py        the online SFT loop -> outputs/results.json + the figures

Every hyper-parameter sits at the top of its file, right after the imports —
shared data/model knobs here, baseline knobs in run_baselines.py, training knobs
in run_sft.py.
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

# Model is a normal HF download (cached after first run). If you're offline or
# rate-limited, export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 before running.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

# --- shared knobs: the model and the stream --------------------------------- #
MODEL_NAME = "LiquidAI/LFM2.5-230M"   # phone-class base model (stays frozen)
SEED = 7
ACTIONS = ("INTERRUPT", "LATER", "ARCHIVE")       # the 3-way attention decision
REGIMES = ("weekday", "on-call", "off-hours")     # three policies, two drifts

# A realistic week, NOT equal thirds: 50% regular weekdays, 20% on-call,
# 30% off-hours (evenings + the weekend) — 30 / 12 / 18 of a 60-item stream.
# The skew is part of the story: RAG's decision store ends up dominated by
# weekday decisions, so it serves stale weekday answers on off-hours queries.
STREAM_LEN = 60
DRIFTS = (30, 42)    # weekday items 1-30 | on-call 31-42 | off-hours 43-60
EVAL_N = 12          # held-out eval items per regime policy
MAX_NEW = 40         # generated tokens per reply (the answer itself is 1-2 tokens)

OUT_DIR = Path("outputs")
BASELINES_JSON = OUT_DIR / "baselines.json"
DATA_OUT = Path("data/inbox_triage.json")   # committed — the Colab notebook fetches it
FIG_DIR = Path("figures")


# --------------------------------------------------------------------------- #
# A tiny dataset: a stream that drifts twice.  The policy is keyword/sender
# driven so a 230M LoRA can pick it up in a few dozen steps; each drift is a
# clean regime change.
# --------------------------------------------------------------------------- #
NAMES = ["Priya", "Marcus", "Lena", "Diego", "Aisha", "Tom", "Yuki", "Sam"]
PROJECTS = ["Atlas", "the Q3 launch", "the billing rewrite", "Project Nomad", "the search revamp"]
PRODUCTS = ["Nimbus", "the mobile app", "SoundOff", "Kettle", "Zephyr"]
INCIDENTS = ["payment latency", "checkout 5xx errors", "a pager alert", "an API outage",
             "elevated error rate", "payment gateway timeouts", "a DB failover"]


def pick(rng: random.Random, options: list[str]) -> str:
    return options[rng.randrange(len(options))]


def gen_item(category: str, rng: random.Random) -> dict:
    """One synthetic inbox item: {subject, snippet}."""
    if category == "mgr_project":
        project = pick(rng, PROJECTS)
        manager = pick(rng, NAMES)
        snippet = pick(rng, [
            f"Can you weigh in before the standup? It's blocking {project}.",
            f"Quick decision needed on {project} — reviewers are waiting on you.",
            f"Are we shipping {project} today? Need your sign-off.",
        ])
        return {"subject": f"Need your call on {project} FROM {manager}", "snippet": snippet}
    if category == "teammate_fyi":
        sender = f"{pick(rng, NAMES)}"
        snippet = pick(rng, [
            "Left a couple of comments on your PR whenever you get a sec.",
            "Shared some notes from the sync — no action needed today.",
            "Thinking about refactoring the utils, curious what you think sometime.",
        ])
        return {"subject": f"fyi / no rush FROM {sender}", "snippet": snippet}
    if category == "calendar_soon":
        who = pick(rng, NAMES)
        return {"subject": f"Starts in 20 min: 1:1 with {who}",
                "snippet": f"Reminder: your meeting with {who} begins soon."}
    if category == "promo":
        sender = f"{pick(rng, PRODUCTS)} Team"
        subject = pick(rng, ["48-hour sale — 40% off", "New features you'll love",
                             "We miss you! Come back for 20% off"])
        return {"subject": f"{subject} FROM {sender}",
                "snippet": "Limited time only. Unsubscribe anytime."}
    if category == "social":
        subject = pick(rng, ["5 people liked your post", "You have 3 new followers",
                             "Someone mentioned you in a comment"])
        return {"subject": subject, "snippet": "Tap to see the activity."}
    if category == "receipt":
        product = pick(rng, PRODUCTS)
        snippet = pick(rng, [
            "Payment of $12.00 received. Thanks for your order.",
            "Your invoice is attached. No action required.",
            "Charge confirmed. View your billing history anytime.",
        ])
        return {"subject": f"Your payment to {product} was successful", "snippet": snippet}
    if category == "monitoring":
        incident = pick(rng, INCIDENTS)
        snippet = pick(rng, [
            f"Automated alert: {incident} detected in production.",
            f"Threshold breached — {incident}. Dashboard link inside.",
            f"{incident} on the checkout service. Auto-generated.",
        ])
        return {"subject": f"[ALERT] {incident}", "snippet": snippet}
    raise ValueError(category)


# The user's latent 3-way policy per regime — the table in the blog's "a stream
# that drifts" section. Several categories flip on each drift: monitoring goes
# ARCHIVE -> INTERRUPT when you go on-call, mgr_project triple-flips
# INTERRUPT -> LATER -> ARCHIVE, and social flips to INTERRUPT off-hours
# (your friends DO deserve a buzz on Saturday).
POLICY = {  #                weekday      on-call      off-hours
    "mgr_project":   ("INTERRUPT", "LATER",     "ARCHIVE"),
    "calendar_soon": ("INTERRUPT", "INTERRUPT", "LATER"),
    "teammate_fyi":  ("LATER",     "ARCHIVE",   "ARCHIVE"),
    "monitoring":    ("ARCHIVE",   "INTERRUPT", "ARCHIVE"),
    "promo":         ("ARCHIVE",   "ARCHIVE",   "LATER"),
    "social":        ("ARCHIVE",   "ARCHIVE",   "INTERRUPT"),
    "receipt":       ("LATER",     "ARCHIVE",   "LATER"),
}


def true_action(category: str, phase: int) -> str:
    """Latent per-regime policy. Phase 1 = weekday, 2 = on-call, 3 = off-hours."""
    return POLICY[category][phase - 1]


def item_block(item: dict) -> str:
    return (f"Subject: {item['subject']}\n"
            f"{item['snippet']}")


def render_prompt(item: dict) -> str:
    return ("Triage this inbox item into one of INTERRUPT (buzz the user now), "
            "LATER (hold it for the digest), or ARCHIVE (never surface it). "
            "Answer with exactly one of INTERRUPT, LATER, or ARCHIVE.\n\n"
            + item_block(item))


def item_prompt(item: dict) -> str:
    """Prefer a pre-rendered prompt (exported / Colab JSON) when present."""
    return item["prompt"] if "prompt" in item else render_prompt(item)


# Category mixes, class-balanced under each regime's policy (10/10/10, 4/4/4,
# 6/6/6 in the stream blocks; 4/4/4 in every eval set) so a tiny LoRA learns
# the mapping, not the class prior. Block lengths follow the 50/20/30 week.
PHASE_SPECS = {
    1: {"mgr_project": 5, "calendar_soon": 5, "teammate_fyi": 5, "receipt": 5,
        "monitoring": 4, "promo": 3, "social": 3},                    # 30 items (10/10/10)
    2: {"calendar_soon": 2, "monitoring": 2, "mgr_project": 4,
        "teammate_fyi": 1, "promo": 1, "social": 1, "receipt": 1},    # 12 items (4/4/4)
    3: {"social": 6, "calendar_soon": 2, "promo": 2, "receipt": 2,
        "mgr_project": 2, "teammate_fyi": 2, "monitoring": 2},        # 18 items (6/6/6)
}
EVAL_SPECS = {  # 12 held-out items per regime, 4/4/4 under that regime's policy
    1: {"mgr_project": 2, "calendar_soon": 2, "teammate_fyi": 2, "receipt": 2,
        "monitoring": 2, "promo": 1, "social": 1},
    2: {"calendar_soon": 2, "monitoring": 2, "mgr_project": 4,
        "teammate_fyi": 1, "promo": 1, "social": 1, "receipt": 1},
    3: {"social": 4, "calendar_soon": 2, "promo": 1, "receipt": 1,
        "mgr_project": 2, "teammate_fyi": 1, "monitoring": 1},
}


def build_slice(mix: dict, phase: int, rng: random.Random) -> list[dict]:
    """Materialise one slice of items from a category mix, labelled and shuffled."""
    items = []
    for category, count in mix.items():
        for _ in range(count):
            item = gen_item(category, rng)
            item["category"] = category   # bookkeeping only — never rendered into the prompt
            item["phase"] = phase
            item["action"] = true_action(category, phase)
            items.append(item)
    rng.shuffle(items)
    return items


def build_stream(rng: random.Random) -> list[dict]:
    """Weekday block, then on-call, then off-hours; labels drift at each boundary."""
    return sum((build_slice(PHASE_SPECS[phase], phase, rng) for phase in (1, 2, 3)), [])


def build_eval(rng: random.Random, phase: int) -> list[dict]:
    return build_slice(EVAL_SPECS[phase], phase, rng)


def phase_of(pos: int) -> int:
    """Regime of stream position `pos` (1-indexed). Position 0 — nothing
    streamed yet — counts as the upcoming weekday regime."""
    if pos <= DRIFTS[0]:
        return 1
    return 2 if pos <= DRIFTS[1] else 3


def recent_demos(history: list[dict], k: int) -> list[tuple[dict, str]]:
    """The k most recent observed decisions, oldest first — the causal ICL
    context (and optional history prepended to the SFT teacher chat)."""
    return [(item, item["action"]) for item in history[-k:]]


def export_dataset(stream: list[dict], evals: dict[int, list[dict]]) -> None:
    """Write the seeded dataset — with prompts pre-rendered — to DATA_OUT.

    The file is committed to the repo, so the Colab notebook fetches the exact
    items the scripts trained on instead of regenerating them at runtime."""
    def enrich(item: dict) -> dict:
        return {**item, "prompt": render_prompt(item)}

    payload = {
        "config": {"seed": SEED, "stream_len": STREAM_LEN, "drifts": list(DRIFTS),
                   "regimes": list(REGIMES), "actions": list(ACTIONS), "eval_n": EVAL_N},
        "stream": [enrich(item) for item in stream],
        "evals": {str(phase): [enrich(item) for item in items]
                  for phase, items in evals.items()},
    }
    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    DATA_OUT.write_text(json.dumps(payload, indent=1) + "\n")


# --------------------------------------------------------------------------- #
# Model helpers
# --------------------------------------------------------------------------- #
def pick_device() -> str:
    forced = os.environ.get("FORCE_DEVICE", "").strip().lower()
    if forced in {"cpu", "mps", "cuda"}:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"    # left-pad so batched generation lines up
    return tok


def load_base_model(device: str):
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype)
    return model.to(device)


def to_model_device(encoding: dict, model) -> dict:
    device = next(model.parameters()).device
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in encoding.items()}


def demo_messages(item: dict, action: str) -> list[dict]:
    return [{"role": "user", "content": item_prompt(item)},
            {"role": "assistant", "content": action}]


def build_msgs(item: dict, demos: list[tuple[dict, str]] | None = None) -> list[dict]:
    """Student / serving chat: bare triage prompt, optional causal ICL demos."""
    messages: list[dict] = []
    for demo_item, demo_action in demos or []:
        messages += demo_messages(demo_item, demo_action)
    messages.append({"role": "user", "content": item_prompt(item)})
    return messages


def build_teacher_msgs(item: dict, expert_action: str,
                       demos: list[tuple[dict, str]] | None = None) -> list[dict]:
    """Teacher chat: same model, privileged with the expert (user) action.

    Shows the user's actual behavior for this item as an in-context demonstration,
    then re-asks the bare triage question so the teacher produces π(·|x, c)
    with c = observed user behavior (used only when DISTILL_BETA > 0).
    Optional `demos` are older causal decisions prepended before that demo.
    """
    messages: list[dict] = []
    for demo_item, demo_action in demos or []:
        messages += demo_messages(demo_item, demo_action)
    messages += demo_messages(item, expert_action)
    messages.append({"role": "user", "content": item_prompt(item)})
    return messages


def make_retriever(store: list[dict]):
    """RAG's index: bag-of-words overlap against a decision history
    (a stand-in for an on-device vector index). `upto` restricts retrieval to
    store[:upto] — the online setting, where only past decisions exist yet."""
    vocab = [set(re.findall(r"\w+", render_prompt(item).lower())) for item in store]

    def retrieve(item: dict, k: int, upto: int | None = None) -> list[tuple[dict, str]]:
        query = set(re.findall(r"\w+", render_prompt(item).lower()))
        pool = range(len(store) if upto is None else min(upto, len(store)))
        ranked = sorted(pool, key=lambda i: -len(query & vocab[i]))
        return [(store[i], store[i]["action"]) for i in ranked[:k]]

    return retrieve


ACTION_RE = re.compile(r"\b(" + "|".join(ACTIONS) + r")\b", re.IGNORECASE)


def parse_action(text: str) -> str:
    match = ACTION_RE.search(text or "")
    return match.group(1).upper() if match else "NONE"


def prompt_tokens(tok, messages: list[dict]) -> int:
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return len(tok(text, add_special_tokens=False)["input_ids"])


def accuracy(items: list[dict], replies: list[str]) -> float:
    hits = sum(parse_action(reply) == item["action"] for item, reply in zip(items, replies))
    return hits / max(len(items), 1)


@torch.inference_mode()
def generate(model, tok, msgs_list: list[list[dict]], *,
             label: str = "gen", batch_size: int = 8, max_new: int = MAX_NEW) -> list[str]:
    """Greedy-decode one reply per chat in msgs_list, batched, with progress prints."""
    model.eval()
    replies: list[str] = []
    for start in range(0, len(msgs_list), batch_size):
        chats = msgs_list[start:start + batch_size]
        texts = [tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                 for chat in chats]
        encoding = to_model_device(
            tok(texts, return_tensors="pt", padding=True, add_special_tokens=False), model)
        output = model.generate(**encoding, max_new_tokens=max_new, do_sample=False,
                                pad_token_id=tok.pad_token_id)
        completions = output[:, encoding["input_ids"].shape[1]:]
        replies += [text.strip()
                    for text in tok.batch_decode(completions, skip_special_tokens=True)]
        print(f"  [{label}] {min(start + batch_size, len(msgs_list))}/{len(msgs_list)}", flush=True)
    return replies
