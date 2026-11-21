#!/usr/bin/env python3
"""Fake HN (Algolia) + fake Ollama on one port, for testing hn_digest.py offline."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

def comment(i, author, text, kids=()):
    return {"id": i, "author": author, "text": text, "children": list(kids)}

ITEM = {
    "id": 1001, "title": "Show HN: A new database", "url": "https://example.com/db",
    "author": "alice", "points": 321, "created_at_i": 1763700000, "text": None,
    "children": [
        comment(2001, "bob", "<p>This is great, but benchmarks &amp; claims need independent checks. See <a href=\"https:&#x2F;&#x2F;example.com&#x2F;bench\" rel=\"nofollow\">link</a></p>",
                [comment(3001, "carol", "I ran them, numbers hold up.", [comment(4001, "bob", "Thanks, that settles it.")]),
                 comment(3002, "dave", "Only on SSDs though.")]),
        comment(2002, "erin", "License is a dealbreaker for me."),
        comment(2003, "frank", None),
    ],
}

def fake_summary():
    return {
        "tldr": "People like the database but want independent benchmarks.",
        "positions": [
            {"title": "Benchmarks hold up", "stance": "support", "summary": "carol reproduced results.",
             "key_arguments": ["Reproduced locally"], "comment_ids": ["3001", "999999"]},
            {"title": "License concerns", "stance": "WEIRD", "summary": "erin dislikes the license.",
             "key_arguments": "Not open enough", "comment_ids": ["2002"]},
        ],
        "agreements": ["Idea is promising"], "disagreements": ["SSD-only performance"], "useful_points": ["Benchmark link"],
    }

class H(BaseHTTPRequestHandler):
    prompts = []
    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path.startswith("/search"):
            self._send({"hits": [
                {"objectID": "1001", "title": ITEM["title"], "url": ITEM["url"], "author": "alice", "points": 321, "num_comments": 40, "created_at_i": 1763700000},
                {"objectID": "1002", "title": "Tiny thread", "url": None, "author": "x", "points": 5, "num_comments": 2, "created_at_i": 1763700100},
            ]})
        elif self.path.startswith("/items/1001"):
            self._send(ITEM)
        else:
            self.send_response(404); self.end_headers()
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        open("/tmp/last_prompt.txt", "w").write(body["messages"][0]["content"])
        assert body["format"] == "json" and body["stream"] is False
        self._send({"message": {"content": json.dumps(fake_summary())}})
    def log_message(self, *a): pass

HTTPServer(("127.0.0.1", 8765), H).serve_forever()
