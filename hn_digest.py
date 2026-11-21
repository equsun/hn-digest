#!/usr/bin/env python3
"""HN Digest: summarize Hacker News comment threads with a local Ollama model.

Pulls the HN front page (free Algolia API), picks the most substantive comments
of each thread, asks Ollama for a structured summary, and writes static JSON
files that site/index.html renders. Python standard library only.

Examples:
  python3 hn_digest.py                 # one pass over the front page
  python3 hn_digest.py --loop 30       # repeat every 30 minutes
  python3 hn_digest.py --id 12345678   # summarize one specific HN item
"""
import argparse
import html
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HN_API = os.environ.get("HN_API", "https://hn.algolia.com/api/v1").rstrip("/")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).parent / "site" / "data"))

PROMPT = """You are summarizing a Hacker News discussion so a busy reader can grasp what people think without reading it.

Story title: {title}
Story link: {url}
{story_text}
Below are the selected comments. Format: [id] (author, N replies) text. Indented lines are replies to the line above.

{comments}

Write a summary of the DISCUSSION (not the article). Be faithful: only use what the comments say, do not invent claims.
Group comments into the 3 to 6 main positions or themes. Mention who is arguing what by author name when useful.
Respond with ONLY a JSON object of exactly this shape:
{{
  "tldr": "2-3 sentences: what the thread is about and the overall mood",
  "positions": [
    {{
      "title": "short name of the position or theme",
      "stance": "support | oppose | mixed | neutral",
      "summary": "1-3 sentences",
      "key_arguments": ["specific argument or evidence given", "..."],
      "comment_ids": ["ids of 1-3 comments that best represent it"]
    }}
  ],
  "agreements": ["points most commenters agree on"],
  "disagreements": ["the main open disputes, with both sides stated"],
  "useful_points": ["corrections, facts, tools, alternatives or links worth knowing"]
}}"""


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def http_json(url, payload=None, timeout=30):
    data = None
    headers = {"User-Agent": "hn-digest/1.0"}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def clean(text):
    """HN comment HTML -> plain text, keeping link targets."""
    if not text:
        return ""
    text = re.sub(r'<a [^>]*href="([^"]+)"[^>]*>.*?</a>', r" \1 ", text, flags=re.S)
    text = re.sub(r"<p>|<br ?/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"[ \t]+", " ", html.unescape(text)).strip()


def count_replies(node):
    return sum(1 + count_replies(c) for c in node.get("children", []) if c.get("text"))


def render_thread(node, depth, max_depth, per_comment, max_replies, ids, budget):
    """Render a comment and its most active replies. Returns list of lines."""
    text = clean(node.get("text"))
    if not text:
        return []
    if len(text) > per_comment:
        text = text[:per_comment].rsplit(" ", 1)[0] + " ..."
    cid = str(node["id"])
    ids.add(cid)
    n = count_replies(node)
    indent = "  " * depth
    lines = [f"{indent}[{cid}] ({node.get('author', '?')}, {n} replies) {text.replace(chr(10), ' ')}"]
    if depth < max_depth:
        kids = [c for c in node.get("children", []) if c.get("text")]
        top = sorted(kids, key=count_replies, reverse=True)[:max_replies]
        keep = {id(k) for k in top}
        for kid in kids:  # preserve original (HN ranked) order
            if id(kid) in keep:
                lines += render_thread(kid, depth + 1, max_depth, per_comment, max_replies, ids, budget)
    return lines


def select_comments(root, budget=24000, per_comment=700, thread_cap=3500):
    """Pick the most substantive comments within a character budget.

    Top-level comments keep HN's ranking order. Within each thread we keep the
    replies with the largest sub-discussions, so back-and-forth is represented
    without letting one giant thread eat the whole budget.
    """
    ids, out, used, used_threads = set(), [], 0, 0
    for top in root.get("children", []):
        lines = render_thread(top, 0, 3, per_comment, 4, ids, budget)
        if not lines:
            continue
        block, size = [], 0
        for ln in lines:
            if size + len(ln) > thread_cap and block:
                break
            block.append(ln)
            size += len(ln) + 1
        if used + size > budget:
            break
        out += block
        used += size
        used_threads += 1
    return "\n".join(out), ids, used_threads


def ollama_summary(model, title, url, story_text, comments, num_ctx, timeout):
    prompt = PROMPT.format(
        title=title,
        url=url or "(none, text post)",
        story_text=f"Post text: {story_text[:1500]}\n" if story_text else "",
        comments=comments,
    )
    resp = http_json(
        f"{OLLAMA_URL}/api/chat",
        {
            "model": model,
            "stream": False,
            "format": "json",
            "options": {"num_ctx": num_ctx, "temperature": 0.2},
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=timeout,
    )
    raw = resp["message"]["content"]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def as_list(v):
    if isinstance(v, str):
        v = [v]
    return [str(x).strip() for x in v if str(x).strip()] if isinstance(v, list) else []


def normalize(raw, valid_ids):
    positions = []
    for p in raw.get("positions", []) if isinstance(raw.get("positions"), list) else []:
        if not isinstance(p, dict) or not p.get("title"):
            continue
        stance = str(p.get("stance", "neutral")).lower().strip()
        positions.append(
            {
                "title": str(p["title"]).strip(),
                "stance": stance if stance in ("support", "oppose", "mixed", "neutral") else "neutral",
                "summary": str(p.get("summary", "")).strip(),
                "key_arguments": as_list(p.get("key_arguments")),
                "comment_ids": [c for c in (str(x).strip("[] ") for x in as_list(p.get("comment_ids"))) if c in valid_ids][:3],
            }
        )
    return {
        "tldr": str(raw.get("tldr", "")).strip(),
        "positions": positions,
        "agreements": as_list(raw.get("agreements")),
        "disagreements": as_list(raw.get("disagreements")),
        "useful_points": as_list(raw.get("useful_points")),
    }


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    tmp.replace(path)


def needs_refresh(cached, num_comments, max_age_h):
    if not cached:
        return True
    grew = num_comments - cached.get("num_comments", 0)
    age_h = (time.time() - cached.get("summarized_at", 0)) / 3600
    return age_h >= max_age_h and (grew >= 25 or grew >= 0.3 * max(cached.get("num_comments", 1), 1))


def summarize_story(hit, args):
    sid = str(hit["objectID"])
    item = http_json(f"{HN_API}/items/{sid}", timeout=60)
    comments, ids, n_threads = select_comments(item, budget=args.budget)
    if not comments:
        raise RuntimeError("no usable comments")
    raw = ollama_summary(
        args.model, item.get("title") or hit.get("title", ""), item.get("url"),
        clean(item.get("text")), comments, args.num_ctx, args.timeout,
    )
    summary = normalize(raw, ids)
    if not summary["tldr"] and not summary["positions"]:
        raise RuntimeError("model returned an empty summary")
    return {
        "id": sid,
        "title": item.get("title") or hit.get("title", ""),
        "url": item.get("url"),
        "author": item.get("author"),
        "points": item.get("points", hit.get("points", 0)),
        "num_comments": hit.get("num_comments", 0),
        "created_at": item.get("created_at_i") or hit.get("created_at_i"),
        "summarized_at": int(time.time()),
        "model": args.model,
        "threads_used": n_threads,
        "summary": summary,
    }


def index_entry(hit, cached):
    s = (cached or {}).get("summary") or {}
    return {
        "id": str(hit["objectID"]),
        "title": hit.get("title", ""),
        "url": hit.get("url"),
        "author": hit.get("author"),
        "points": hit.get("points", 0),
        "num_comments": hit.get("num_comments", 0),
        "created_at": hit.get("created_at_i"),
        "has_summary": bool(cached),
        "tldr": s.get("tldr", ""),
    }


FEEDS = ("front", "new", "ask", "show")


def feed_url(feed, limit):
    since = int(time.time()) - 2 * 86400  # ask/show: last 2 days, ranked by points
    endpoint, params = {
        "front": ("search", {"tags": "front_page"}),
        "new": ("search_by_date", {"tags": "story"}),
        "ask": ("search", {"tags": "ask_hn", "numericFilters": f"created_at_i>{since}"}),
        "show": ("search", {"tags": "show_hn", "numericFilters": f"created_at_i>{since}"}),
    }[feed]
    return f"{HN_API}/{endpoint}?" + urllib.parse.urlencode(dict(params, hitsPerPage=limit))


def run_once(args):
    items_dir = DATA_DIR / "items"
    if args.id:
        feeds = {None: [{"objectID": args.id, "num_comments": 10 ** 6}]}
    else:
        feeds = {}
        for name in (n.strip() for n in args.feeds.split(",") if n.strip()):
            if name not in FEEDS:
                log(f"unknown feed '{name}' (choose from {', '.join(FEEDS)})")
                continue
            try:
                feeds[name] = http_json(feed_url(name, args.limit), timeout=30)["hits"]
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                log(f"could not load feed '{name}': {e}")
    done = 0
    for name, hits in feeds.items():
        entries = []
        for hit in hits:
            sid = str(hit["objectID"])
            path = items_dir / f"{sid}.json"
            cached = json.loads(path.read_text()) if path.exists() else None
            nc = hit.get("num_comments") or 0
            if nc >= args.min_comments and needs_refresh(cached, nc, args.refresh_hours):
                try:
                    log(f"[{name}] summarizing {sid} ({nc} comments): {hit.get('title', '')[:60]}")
                    cached = summarize_story(hit, args)
                    write_json(path, cached)
                    done += 1
                except Exception as e:  # keep going, one bad thread must not stop the run
                    log(f"  failed {sid}: {e}")
            if name:
                entries.append(index_entry(hit, cached))
                write_json(DATA_DIR / "feeds" / f"{name}.json", {"updated_at": int(time.time()), "stories": entries})
    # drop summaries nobody can reach anymore
    if items_dir.exists():
        cutoff = time.time() - args.keep_days * 86400
        for f in items_dir.glob("*.json"):
            if f.stat().st_mtime < cutoff:
                f.unlink()
    log(f"pass complete: {done} new/updated summaries")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default {DEFAULT_MODEL})")
    ap.add_argument("--limit", type=int, default=30, help="stories to consider per feed")
    ap.add_argument("--feeds", default="front,new,ask,show", help="comma list of: front,new,ask,show (default all)")
    ap.add_argument("--min-comments", type=int, default=15, help="skip threads smaller than this")
    ap.add_argument("--budget", type=int, default=24000, help="max characters of comments sent to the model")
    ap.add_argument("--num-ctx", type=int, default=12288, help="Ollama context window in tokens")
    ap.add_argument("--timeout", type=int, default=900, help="seconds to wait for one summary")
    ap.add_argument("--refresh-hours", type=float, default=6, help="re-summarize grown threads after this long")
    ap.add_argument("--keep-days", type=int, default=14, help="delete summaries older than this")
    ap.add_argument("--loop", type=float, metavar="MIN", help="repeat every MIN minutes")
    ap.add_argument("--id", help="summarize one specific HN item id and exit")
    ap.add_argument("--after", metavar="CMD", help="shell command to run after each pass, e.g. 'npx wrangler deploy'")
    args = ap.parse_args()

    while True:
        try:
            run_once(args)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            log(f"network error: {e}")
        if args.after:
            log(f"running: {args.after}")
            rc = subprocess.run(args.after, shell=True, cwd=Path(__file__).parent).returncode
            if rc:
                log(f"  --after command exited with {rc}")
        if not args.loop or args.id:
            break
        time.sleep(args.loop * 60)


if __name__ == "__main__":
    main()
