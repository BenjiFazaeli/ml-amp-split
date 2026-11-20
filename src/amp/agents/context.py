#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Keeping the request inside the provider's limits: the image budget every model has, and
stage-by-stage elision for models with a known context window."""

from __future__ import annotations

import ast
import base64
import json
import re

IMAGE_LONG_SIDE = 1280
MAX_IMAGES = 80
MAX_IMAGE_B64 = 18_000_000
PROMPT_BUDGET_FRACTION = 0.70
CHARS_PER_TOKEN = 3.0
KEEP_NEWEST = 6
LONG_TEXT = 2000
KEEP_TEXT = 600
LONG_INPUT = 1500
KEEP_INPUT = 500
OVERFLOW_SHRINK = 0.8
REASONING_ELIDED = "[earlier reasoning elided to fit the context window]"
RESULT_ELIDED = ("[tool result elided to fit the context window; view_image shows saved images "
                 "again]")
REFUSED_NOTE = ("[image not attached: the API refused it in this result; view_image shows it "
                "again]")
STAGES = ("thinking", "tool_text", "tool_input", "assistant_text", "images", "tool_results")

# amp-split patch: --trim-select outcome. At a cut (trim_floor set) fixed errors and superseded
# state printouts go first, then reasoning as before; these three go on to the floor together,
# and the later stages open only while the request is over the budget, as with the floor alone.
SELECTS = ("outcome",)
OUTCOME_STAGES = ("resolved_errors", "superseded") + STAGES
DEEPEN = ("resolved_errors", "superseded", "thinking")
RESOLVE_TURNS = 3        # an error is fixed when a python call within this many turns is not one
STATE_QUERIES = ("budget", "status", "objects")     # read-only Episode methods, called bare
CLOCK_RE = re.compile(r"\bep\.(?:budget|status)\(")
OBJECTS_RE = re.compile(r"\bep\.objects\(")
SAFE_CALLS = frozenset({"print", "len", "round", "str", "repr", "sorted", "list", "dict", "tuple",
                        "int", "float", "bool", "min", "max", "abs", "enumerate", "zip", "range",
                        "json.dumps", "pprint.pprint", "np.round"})
SAFE_METHODS = frozenset({"round", "tolist", "items", "keys", "values", "get"})
SAFE_IMPORTS = frozenset({"json", "pprint", "numpy", "math"})
ACTING = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef, ast.With,
          ast.AsyncWith, ast.Try, ast.While, ast.Delete, ast.Global, ast.Nonlocal, ast.AugAssign,
          ast.AnnAssign, ast.NamedExpr, ast.Await, ast.Yield, ast.YieldFrom, ast.Raise)
STEP_RE = re.compile(r"\[step (\d+) of \d+")
FOOTER_RE = re.compile(r"(?:^|\n)(\[step \S+ of [^\n]*\])\s*$")  # the text loop's budget_note
EXC_RE = re.compile(r"[A-Za-z_][\w.]*(?::.*)?$")
EXC_CHARS = 200
FAILED_CODE = "# [code of a failed call, elided to fit the context window]"
FAILED_NOTE = "[failed call elided to fit the context window: {exc}{sep}{fix} ran without error.]"
SUPERSEDED_NOTE = "[result elided to fit the context window: a later call printed {q} again]"
# The vdm and vdm_strict coders have no view_image (tools_for), so with a selection their
# placeholder names what they do have; with the floor alone it stays RESULT_ELIDED.
RESULT_ELIDED_BY_VISION = {
    "vdm": ("[tool result elided to fit the context window; re-run its code to print it again, "
            "or show() a saved image's path to ask the vision model about it again]"),
    "vdm_strict": ("[tool result elided to fit the context window; re-run its code to print it "
                   "again]"),
}


def _text(text: str) -> dict:
    return {"type": "text", "text": text}


def footer(text: str) -> str:
    """The budget note a tool result's text ends with ("\\n[step S of M, ...]"), or "": a
    collapsed result keeps it, so its step and time stay in the history."""
    m = FOOTER_RE.search(text)
    return "\n" + m.group(1) if m else ""


def exception_line(text: str) -> str:
    """The exception a python result's text names ("KeyError: 'mug'"): the last unindented line
    of its error part that starts with a name, skipping the harness's bracketed notes."""
    part = text[text.rfind("error:\n") + len("error:\n"):] if "error:\n" in text else text
    lines = [ln.strip() for ln in part.splitlines() if ln.strip() and not ln[0].isspace()]
    named = [ln for ln in lines if EXC_RE.match(ln) and not ln.startswith("Traceback")]
    line = (named or [ln for ln in lines if not ln.startswith("[")] or ["an error"])[-1]
    return line if len(line) <= EXC_CHARS else line[:EXC_CHARS - 3] + "..."


def failed_note(exc: str, later: int) -> str:
    """The line a failed call's result begins with once collapsed; later: the turns from it to
    the python call that ran without an error."""
    fix = "The next python call" if later == 1 else f"A python call {later} turns later"
    return FAILED_NOTE.format(exc=exc, sep=" " if exc.endswith((".", "!", "?")) else ". ",
                              fix=fix)


def state_query(code: str) -> tuple[frozenset, str] | None:
    """(the Episode state queries, the code's AST dump) when a python call's code is a pure
    state query: it calls only ep.budget(), ep.status() and ep.objects() without arguments,
    SAFE_CALLS and SAFE_METHODS, assigns only plain names, imports only SAFE_IMPORTS and reads
    no kernel variable but ep and those modules. None for any other code. A conservative
    reading: code that might act, or print something a variable chose, is never one."""
    try:
        tree = ast.parse(code or "")
    except (SyntaxError, ValueError):
        return None
    found = set()
    known = {"ep", "np", *SAFE_IMPORTS, *(n for n in SAFE_CALLS if "." not in n)}
    loads = set()
    for node in ast.walk(tree):
        if isinstance(node, ACTING):
            return None
        if isinstance(node, ast.Name):
            (loads if isinstance(node.ctx, ast.Load) else known).add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [node.module or ""] if isinstance(node, ast.ImportFrom) else \
                [a.name for a in node.names]
            if not all(m.split(".")[0] in SAFE_IMPORTS for m in mods):
                return None
            known |= {a.asname or a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.Assign):
            if not all(isinstance(t, ast.Name) for t in node.targets):
                return None
        elif isinstance(node, (ast.For, ast.comprehension)):
            if not isinstance(node.target, (ast.Name, ast.Tuple)) or any(
                    not isinstance(e, ast.Name) for e in getattr(node.target, "elts", [])):
                return None
        elif isinstance(node, ast.Call):
            f = node.func
            owner = f.value.id if isinstance(f, ast.Attribute) and \
                isinstance(f.value, ast.Name) else None
            if owner == "ep":
                if f.attr not in STATE_QUERIES or node.args or node.keywords:
                    return None
                found.add(f.attr)
            elif isinstance(f, ast.Name):
                if f.id not in SAFE_CALLS:
                    return None
            elif not isinstance(f, ast.Attribute):
                return None
            elif f"{owner}.{f.attr}" not in SAFE_CALLS and (
                    f.attr not in SAFE_METHODS or owner in ("rh", "show")):
                return None
    if not found or loads - known:
        return None
    return frozenset(found), ast.dump(tree)


def python_calls(messages: list[dict]) -> list[dict]:
    """Every python call answered in the message after it, oldest first: its turn (the ordinal
    of its assistant message), that message's index and content, the tool_use and how many the
    message has, the user content holding the result and its index there, the result, its code,
    the step its footer ([step S of M, ...]) reports and, for a pure state query, state_query's
    (queries, AST dump)."""
    out = []
    turn = 0
    for j, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        turn += 1
        content = m.get("content")
        nxt = messages[j + 1] if j + 1 < len(messages) else {}
        reply = nxt.get("content") if nxt.get("role") == "user" else None
        if not isinstance(content, list) or not isinstance(reply, list):
            continue
        uses = [b for b in content if b.get("type") == "tool_use"]
        for u in uses:
            k = next((k for k, r in enumerate(reply) if r.get("type") == "tool_result"
                      and r.get("tool_use_id") == u.get("id")), None)
            if u.get("name") != "python" or k is None or \
                    not isinstance(reply[k].get("content"), list):
                continue
            inp = u.get("input")
            code = str(inp.get("code") or "") if isinstance(inp, dict) else ""
            texts = [b.get("text") or "" for b in reply[k]["content"] if b.get("type") == "text"]
            steps = STEP_RE.findall(texts[-1]) if texts else []
            out.append({"turn": turn, "at": j, "content": content, "use": u, "uses": len(uses),
                        "reply": reply, "index": k, "result": reply[k], "code": code,
                        "step": int(steps[-1]) if steps else None,
                        "query": state_query(code)})
    return out


def image_pointer(label: str) -> dict:
    """The text that replaces an image dropped from the context; label is its caption block."""
    path = label.split(":")[0].removeprefix("image ").strip() if label else "?"
    return {"type": "text", "text": f"[{path}: no longer attached to keep the request small; "
                                    "view_image shows it again]"}


def image_size(data: str) -> tuple[int, int] | None:
    """(w, h) from a base64 PNG or JPEG header; None when unreadable."""
    for chunk in (data[:65536], data):
        try:
            raw = base64.b64decode(chunk[: len(chunk) // 4 * 4])
        except ValueError:
            return None
        if raw[:8] == b"\x89PNG\r\n\x1a\n" and len(raw) >= 24:
            return int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")
        if raw[:2] == b"\xff\xd8":
            i = 2
            while i + 9 <= len(raw) and raw[i] == 0xFF:
                kind = raw[i + 1]
                if 0xC0 <= kind <= 0xCF and kind not in (0xC4, 0xC8, 0xCC):
                    return (int.from_bytes(raw[i + 7:i + 9], "big"),
                            int.from_bytes(raw[i + 5:i + 7], "big"))
                i += 2 + int.from_bytes(raw[i + 2:i + 4], "big")
            if chunk is data:
                return None
            continue
        return None
    return None


def shorten(text: str, keep: int) -> str:
    return (f"{text[:keep]}\n...[{len(text) - 2 * keep} characters elided to fit the context "
            f"window]...\n{text[-keep:]}")


def shorten_input(v, changed: list):
    """v with every string longer than LONG_INPUT shortened; changed[0] set when one was."""
    if isinstance(v, str) and len(v) > LONG_INPUT:
        changed[0] = True
        return shorten(v, KEEP_INPUT)
    if isinstance(v, dict):
        return {k: shorten_input(x, changed) for k, x in v.items()}
    if isinstance(v, list):
        return [shorten_input(x, changed) for x in v]
    return v


class ContextWindow:
    """The images in the history (oldest first) and, given context_tokens, the token budget."""

    def __init__(self, context_tokens: int | None = None, image_cost=None,
                 max_images: int = MAX_IMAGES, max_image_b64: int = MAX_IMAGE_B64,
                 trim_floor: float | None = None, select: str | None = None,
                 vision: str | None = None):
        self.context_tokens = context_tokens
        # amp-split patch: once fit has to elide, it cuts to trim_floor x the window instead of
        # to the budget, so the history is append-only again for many turns (vLLM's prefix cache
        # hits) rather than edited on almost every turn. None: the budget, as before.
        if trim_floor is not None and not 0 < float(trim_floor) < PROMPT_BUDGET_FRACTION:
            raise ValueError(f"trim_floor must be above 0 and below {PROMPT_BUDGET_FRACTION}, "
                             f"not {trim_floor}")
        self.trim_floor = None if trim_floor is None else float(trim_floor)
        # amp-split patch: select "outcome" chooses what that cut takes (OUTCOME_STAGES); vision
        # (the harness configuration's) words its placeholders. None: the stages, as before.
        if select is not None and select not in SELECTS:
            raise ValueError(f"select must be one of {', '.join(SELECTS)}, not {select!r}")
        if select is not None and self.trim_floor is None:
            raise ValueError("select needs trim_floor: it chooses what a cut to the floor takes")
        self.select = select
        self.result_elided = RESULT_ELIDED_BY_VISION.get(vision or "image", RESULT_ELIDED) \
            if select else RESULT_ELIDED
        self.image_cost = image_cost
        self.max_images, self.max_image_b64 = max_images, max_image_b64
        self.budget_fraction = PROMPT_BUDGET_FRACTION
        self.token_ratio = 1.0
        self.images: list[tuple[list, dict, str, int]] = []
        self._image_tokens: dict[int, tuple[int, str, int]] = {}

    def register(self, content: list) -> None:
        for b in content:
            if b.get("type") == "image":
                self.images.append((content, b, "", len(b["source"]["data"])))
            elif b.get("type") == "tool_result" and isinstance(b.get("content"), list):
                label = ""
                for c in b["content"]:
                    if c.get("type") == "text":
                        label = c["text"]
                    elif c.get("type") == "image":
                        self.images.append((b["content"], c, label, len(c["source"]["data"])))

    def trim_images(self) -> int:
        """Replace the oldest images by pointers once a cap is near; how many were."""
        if len(self.images) <= self.max_images and \
                sum(n for *_, n in self.images) <= self.max_image_b64:
            return 0
        dropped = 0
        while self.images and (len(self.images) > self.max_images * 3 // 4
                               or sum(n for *_, n in self.images) > self.max_image_b64 * 3 // 4):
            container, block, label, _ = self.images.pop(0)
            for i, c in enumerate(container):
                if c is block:
                    container[i] = image_pointer(label)
            dropped += 1
        return dropped

    def strip_newest(self, messages: list[dict]) -> int:
        """Replace the image blocks of the newest user turn by a note; how many there were."""
        if not messages or messages[-1].get("role") != "user":
            return 0
        removed = set()

        def strip(blocks):
            for i, b in enumerate(blocks):
                if b.get("type") == "image":
                    removed.add(id(b))
                    blocks[i] = {"type": "text", "text": REFUSED_NOTE}
                elif b.get("type") == "tool_result" and isinstance(b.get("content"), list):
                    strip(b["content"])

        content = messages[-1].get("content")
        if isinstance(content, list):
            strip(content)
        self.images = [x for x in self.images if id(x[1]) not in removed]
        return len(removed)

    def image_tokens(self, block: dict) -> int:
        data = block.get("source", {}).get("data") or ""
        key = id(block)
        hit = self._image_tokens.get(key)
        if hit and hit[0] == len(data) and hit[1] == data[-64:]:
            return hit[2]
        w, h = image_size(data) or (IMAGE_LONG_SIDE, IMAGE_LONG_SIDE)
        n = int(self.image_cost(w, h)) if self.image_cost else w * h // 750
        self._image_tokens[key] = (len(data), data[-64:], n)
        return n

    def block_tokens(self, b: dict) -> float:
        kind = b.get("type")
        if kind == "image":
            return self.image_tokens(b)
        if kind == "tool_result":
            c = b.get("content")
            if isinstance(c, str):
                return len(c) / CHARS_PER_TOKEN
            return sum(self.block_tokens(x) for x in c or [])
        if kind == "tool_use":
            n = len(json.dumps(b.get("input") or {})) + len(b.get("name") or "")
            return n / CHARS_PER_TOKEN
        return len(b.get("text") or b.get("thinking") or "") / CHARS_PER_TOKEN

    def message_tokens(self, m: dict) -> float:
        c = m.get("content")
        if isinstance(c, str):
            return len(c) / CHARS_PER_TOKEN
        return sum(self.block_tokens(b) for b in c or [])

    def estimate(self, system: list[dict], tools: list[dict], messages: list[dict]) -> int:
        """The request's size in tokens: chars / 3 for text, image_cost for images."""
        fixed = sum(len(b.get("text") or "") for b in system) + len(json.dumps(tools))
        return round(fixed / CHARS_PER_TOKEN + sum(self.message_tokens(m) for m in messages))

    def calibrate(self, estimated: int, actual: int | None, overflow: bool = False) -> None:
        """Learn the server/estimate ratio from a reply's prompt tokens or an overflow's count."""
        if not (self.context_tokens and estimated and actual):
            return
        ratio = actual / estimated
        self.token_ratio = max(self.token_ratio, ratio) if overflow else max(1.0, ratio)

    def shrink(self) -> None:
        self.budget_fraction *= OVERFLOW_SHRINK

    def fit(self, system: list[dict], tools: list[dict], messages: list[dict]) -> dict | None:
        """Elide old context, stage by stage and oldest first, until the request fits
        budget_fraction of the window; the first user message and the newest KEEP_NEWEST messages
        are never touched. With trim_floor (amp-split patch) a fit that has to elide goes on to
        trim_floor x the window (never above the budget) within the stages it uses, but opens no
        further stage once the request fits the budget. With select "outcome" (amp-split patch)
        that cut first collapses fixed errors and superseded state printouts, then reasoning,
        and never touches the newest clock and object printouts (pins)."""
        if not self.context_tokens:
            return None
        budget = self.budget_fraction * self.context_tokens
        ratio = self.token_ratio
        before = total = self.estimate(system, tools, messages) * ratio
        if total <= budget:
            return None
        target = budget if self.trim_floor is None else \
            min(budget, self.trim_floor * self.context_tokens)
        hi = max(1, len(messages) - KEEP_NEWEST)
        old = messages[1:hi]
        calls = python_calls(messages) if self.select else []
        pins = self.pins(calls)

        def blocks(role):
            for m in old:
                if m.get("role") == role and isinstance(m.get("content"), list):
                    yield m["content"]

        def results():
            for cont in blocks("user"):
                for i, r in enumerate(cont):
                    if r.get("type") == "tool_result" and isinstance(r.get("content"), list) \
                            and id(r) not in pins:
                        yield cont, i, r

        def resolved_errors():
            for n, c in enumerate(calls):
                r = c["result"]
                if not r.get("is_error") or c["uses"] != 1 or c["at"] + 1 >= hi or \
                        c["use"].get("input") == {"code": FAILED_CODE}:
                    continue
                fix = None
                for d in calls[n + 1:]:
                    if d["turn"] - c["turn"] > RESOLVE_TURNS:
                        break
                    if not d["result"].get("is_error"):
                        fix = d
                        break
                prev = calls[n - 1]["step"] if n else 0     # every episode starts at step 0
                if fix is None or prev is None or c["step"] != prev:  # unfixed, or it moved
                    continue
                edits = []
                for i, b in enumerate(c["content"]):
                    if b.get("type") == "thinking" and b.get("thinking") != REASONING_ELIDED:
                        edits.append((c["content"], i, {**b, "thinking": REASONING_ELIDED}))
                    elif b is c["use"]:
                        edits.append((c["content"], i, {**b, "input": {"code": FAILED_CODE}}))
                rc = r["content"]
                text_first = bool(rc) and rc[0].get("type") == "text"
                head = (rc[0].get("text") or "") if text_first else ""
                note = failed_note(exception_line(head), fix["turn"] - c["turn"])
                rest = rc[1:] if text_first else rc
                edits.append((c["reply"], c["index"],
                              {**r, "content": [_text(note + footer(head)), *rest]}))
                yield edits

        def superseded():
            for n, c in enumerate(calls):
                r = c["result"]
                if r.get("is_error") or c["query"] is None or id(r) in pins or \
                        c["at"] + 1 >= hi:
                    continue
                if any(d["query"] == c["query"] and not d["result"].get("is_error")
                       for d in calls[n + 1:]):          # the same code, run again later
                    q = ", ".join(f"ep.{x}()" for x in STATE_QUERIES if x in c["query"][0])
                    texts = [b.get("text") or "" for b in r["content"] if b.get("type") == "text"]
                    note = SUPERSEDED_NOTE.format(q=q) + footer(texts[-1] if texts else "")
                    yield [(c["reply"], c["index"], {**r, "content": [_text(note)]})]

        def thinking():
            for cont in blocks("assistant"):
                for i, b in enumerate(cont):
                    if b.get("type") == "thinking":
                        yield cont, i, {**b, "thinking": REASONING_ELIDED}

        def tool_text():
            for _, _, r in results():
                for i, b in enumerate(r["content"]):
                    t = b.get("text") or ""
                    if b.get("type") == "text" and len(t) > LONG_TEXT:
                        yield r["content"], i, {**b, "text": shorten(t, KEEP_TEXT)}

        def tool_input():
            for cont in blocks("assistant"):
                for i, b in enumerate(cont):
                    changed = [False]
                    if b.get("type") == "tool_use":
                        inp = shorten_input(b.get("input"), changed)
                        if changed[0]:
                            yield cont, i, {**b, "input": inp}

        def assistant_text():
            for cont in blocks("assistant"):
                for i, b in enumerate(cont):
                    t = b.get("text") or ""
                    if b.get("type") == "text" and len(t) > LONG_TEXT:
                        yield cont, i, {**b, "text": shorten(t, KEEP_TEXT)}

        def images():
            for top in blocks("user"):
                for cont in [top] + [r["content"] for r in top if r.get("type") == "tool_result"
                                     and isinstance(r.get("content"), list)
                                     and id(r) not in pins]:
                    label = ""
                    for i, b in enumerate(cont):
                        if b.get("type") == "text":
                            label = b.get("text") or ""
                        elif b.get("type") == "image":
                            yield cont, i, image_pointer(label)

        def whole_results():
            for cont, i, r in results():
                yield cont, i, {**r, "content": [{"type": "text", "text": self.result_elided}]}

        def image_ids(b):
            if b.get("type") == "image":
                return {id(b)}
            if isinstance(b.get("content"), list):
                return set().union(*(image_ids(x) for x in b["content"]))
            return set()

        stages = dict(zip(OUTCOME_STAGES, (resolved_errors, superseded, thinking, tool_text,
                                           tool_input, assistant_text, images, whole_results)))
        # amp-split patch: an item of a stage is one edit (cont, i, repl) or, for the selection's
        # stages, a list of them made together (a failed call's reasoning, code and result).
        names, deepen = (OUTCOME_STAGES, DEEPEN) if self.select else (STAGES, ())
        counts = dict.fromkeys(names, 0)
        freed = dict.fromkeys(names, 0.0)
        dropped: set[int] = set()
        for name in names:
            # amp-split patch: the floor deepens a stage, never adds one; with a selection its
            # first three stages (DEEPEN) go on to the floor together
            if total <= (target if name in deepen else budget):
                break
            for item in stages[name]():
                edits = item if isinstance(item, list) else [item]
                gain = sum(self.block_tokens(cont[i]) - self.block_tokens(repl)
                           for cont, i, repl in edits) * ratio
                if gain <= 0:
                    continue
                for cont, i, repl in edits:
                    dropped |= image_ids(cont[i]) - image_ids(repl)
                    cont[i] = repl
                total -= gain
                counts[name] += 1
                freed[name] += gain
                if total <= target:
                    break
        if dropped:
            self.images = [x for x in self.images if id(x[1]) not in dropped]
            for k in dropped:
                self._image_tokens.pop(k, None)
        after = self.estimate(system, tools, messages) * ratio
        out = {**counts, "tokens_before": round(before), "tokens_after": round(after),
               "budget_tokens": round(budget), "fraction": round(self.budget_fraction, 4),
               "target_tokens": round(target), "trim_floor": self.trim_floor,
               "token_ratio": round(ratio, 3), "context_tokens": self.context_tokens,
               "fits": after <= budget}
        if self.select:
            out.update(select=self.select, pinned=len(pins),
                       freed_tokens={k: round(v) for k, v in freed.items() if v})
        return out

    @staticmethod
    def pins(calls: list[dict]) -> set[int]:
        """The ids of the tool results a selection never elides: the newest python result
        without an error whose code calls ep.budget() or ep.status(), and likewise for
        ep.objects(). An earlier episode lost its clock printouts to the trimmer."""
        pins = set()
        for pattern in (CLOCK_RE, OBJECTS_RE):
            c = next((c for c in reversed(calls) if not c["result"].get("is_error")
                      and pattern.search(c["code"])), None)
            if c is not None:
                pins.add(id(c["result"]))
        return pins
