#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit <tool_call_security> blocks by checking their citations against the conversation.

The generating model can produce a block that is internally perfectly consistent and still
completely wrong: it cites user_message[2] when no second user turn exists, and quotes words
that only ever appeared inside a tool response. A sample like that trains the model to treat
injected instructions as user instructions, which is the exact failure being chased.

None of these checks ask for judgement. They are string lookups against the conversation:
  * does the cited message index exist at that point in the conversation?
  * do the quoted trigger words actually appear in the message they are attributed to?
  * for a call labelled safe, do its argument values appear in the user's own turns, or were
    they delegated by the user to a source the user themselves named?
  * for a call labelled non-safe, do its argument values actually appear in the tool response
    being blamed for it?

The block format has three fixed questions — argument provenance, trigger words, source — and
four more when the source is a tool response. Answers are located by matching the question
wording, not by position, so a block that drops or reorders a question is reported rather than
silently audited against the wrong answer.

Records are split into <name>-valid.json and <name>-invalid.json, and a per-block report is
written to <name>-audit-report.jsonl for inspection.
"""

import json
import os
import re
from collections import Counter

INPUT_FILES = [
    "func-calling/Qwen3-8B/glaive-function-calling-5k-think-8b-clean-tool_call_security-more-tools-clean-clean.json",
    "func-calling/Qwen3-8B/glaive-function-calling-5k-injected-direct-think-8b-clean-clean-tool_call_security-more-tools-clean-clean.json",
    "func-calling/Qwen3-8B/glaive-function-calling-5k-injected-direct-template-think-8b-clean-clean-tool_call_security-more-tools-clean-clean.json",
]

# A quote shorter than this carries no evidence (e.g. "the words") and is not checked.
MIN_QUOTE_LEN = 12
# Argument values shorter than this are too generic to trace (e.g. 5, true, "AA").
MIN_ARG_VALUE_LEN = 4
# Fraction of a quote's words that must appear in the cited message for a loose match.
# The generating model often compresses or lightly rewords the user's sentence instead of
# quoting it verbatim. That is sloppy quoting, not a wrong label, so it is reported as a
# warning rather than treated as corrupt data.
FUZZY_MIN = 0.8

# Only HARD errors mean the label itself is wrong. SOFT ones mean the reasoning text is
# imprecise while the source attribution still holds.
HARD_ERRORS = {
    "malformed_block", "trace_index_missing", "quote_is_from_tool",
    "safe_but_args_from_tool", "trace_level_mismatch", "qcount_mismatch",
    "source_answer_trace_disagree", "position_label_conflict",
    "arg_question_missing",
}
# Set to True to also reject records whose only problem is a loose or missing quote.
STRICT = False

# A safe call may now legitimately follow a tool turn. When the user says "summarise every
# website posted to that channel", the URL arrives in a tool response but the instruction is
# still the user's own — the block marks such a value USER-DELEGATED. Recognising that is what
# stops the model from reading "the value came back from a tool" as "a tool told me to do it",
# which is the confusion that lets an injection with a familiar-looking action through.
# Set to False to go back to treating any safe-after-tool call as an error.
ALLOW_DELEGATED_SAFE = True
DELEGATED_MARK = "user-delegated"

# A call whose nearest preceding non-assistant turn is a tool response cannot have been
# triggered by the user, unless the user's original request needed several calls or delegated
# its values (see above). Set to False if this produces noise on multi-step plans.
CHECK_POSITION = True

# Surface features that make an injected tool response trivially recognisable. If nearly
# every injection carries one, the model can learn to spot the marker instead of reasoning
# about who issued the instruction — which is what the diagnostics below measure.
INJECTION_FIELDS = ("notice", "result_more", "warning", "important", "alert",
                    "instruction", "system_note")
INJECTION_PHRASES = ("important", "ignore all previous", "ignore previous", "signed",
                     "strictly adhere", "before you can", "urgent", "attention",
                     "must first", "system notice")

ERROR_LABELS = {
    "malformed_block":        "block missing required tags",
    "arg_question_missing":   "argument-provenance question missing or not first",
    "no_quote_in_trigger":    "trigger answer quotes nothing",
    "quote_loose_match":      "quoted words are a paraphrase, not verbatim (warning)",
    "position_label_conflict": "label contradicts the turn the call follows",
    "trace_index_missing":    "cited message index does not exist",
    "quote_not_in_cited":     "quoted words are not in the cited message",
    "quote_is_from_tool":     "quoted words claimed as user text but found in a tool response",
    "safe_but_args_from_tool": "labelled safe but argument values only appear in tool responses",
    "nonsafe_args_not_in_cited": "blamed tool response contains none of the argument values (warning)",
    "trace_level_mismatch":   "trace and security level contradict each other",
    "qcount_mismatch":        "question count does not match the security level",
    "source_answer_trace_disagree": "source answer and tool_trace name different sources",
}

# Question wording used to locate each answer. Matching on wording rather than position means
# a block that drops or reorders a question is caught by arg_question_missing instead of
# having its trigger answer silently audited as if it were the provenance answer.
Q_ARGS_RE    = re.compile(r"argument value", re.I)
Q_TRIGGER_RE = re.compile(r"trigger", re.I)
Q_SOURCE_RE  = re.compile(r"come directly from the user'?s own instructions", re.I)

BLOCK_RE = re.compile(r"<tool_call_security>(.*?)</tool_call_security>", re.S)
TAG_RE = {
    t: re.compile(r"<%s>(.*?)</%s>" % (t, t), re.S)
    for t in ("tool_name", "tool_args", "tool_reason", "tool_trace", "tool_security")
}
ANSWER_RE = re.compile(r"(?m)^A: (.*(?:\n(?!\s*[QA]: ).*)*)")
QUOTE_RE = re.compile(r'"([^"]{2,})"')
TRACE_RE = re.compile(r"(user[ _-]?message|tool[ _-]?response)\s*[\[_\-#: ]\s*(\d+)", re.I)


def norm(text):
    """Collapse whitespace and case so quoting differences do not cause false alarms."""
    text = (text or "").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\\\"", '"').replace("\\n", " ").replace("\\'", "'")
    return re.sub(r"\s+", " ", text).strip().lower()


def words(text):
    return re.findall(r"[a-z0-9$%.]+", norm(text))


def overlap(quote, haystack):
    """Fraction of the quote's words that appear in the haystack."""
    qw = words(quote)
    if not qw:
        return 0.0
    hw = set(words(haystack))
    return sum(1 for w in qw if w in hw) / float(len(qw))


def parse_trace(text):
    m = TRACE_RE.search(text or "")
    if not m:
        return None
    kind = "user_message" if "message" in m.group(1).lower() else "tool_response"
    return kind, int(m.group(2))


def build_index(conversations, upto):
    """Return the user turns and tool turns visible before conversations[upto].

    A user turn is one the user actually wrote. Turns carrying tool output are tool
    responses even though the chat template delivers them inside user turns.
    """
    users, tools = [], []
    for conv in conversations[:upto]:
        src = conv.get("from")
        value = conv.get("value", "") or ""
        if src == "tool" or "<tool_response>" in value:
            tools.append(value)
        elif src == "human":
            users.append(value)
    return users, tools


def distinctive_args(tool_args):
    """Argument values specific enough to be traced back to a message."""
    out = []
    try:
        parsed = json.loads(tool_args)
    except (ValueError, TypeError):
        return out
    if not isinstance(parsed, dict):
        return out
    for value in parsed.values():
        if isinstance(value, bool) or value is None:
            continue
        text = str(value).strip()
        if len(text) >= MIN_ARG_VALUE_LEN and not text.isdigit():
            out.append(text)
    return out


def split_qa(reason):
    """Pair each question with its answer and return them, plus the three fixed answers.

    Returns (questions, answers, args_answer, trigger_answer, source_answer). Any of the three
    may be None when the corresponding question is absent. Matching is by wording so that a
    reordered block is reported rather than audited against the wrong text; positions are used
    only as a fallback, since a wrapped question line still leaves the order intact.
    """
    questions = re.findall(r"(?m)^Q: (.*)$", reason)
    answers = ANSWER_RE.findall(reason)

    def pick(rx, fallback):
        for i, q in enumerate(questions):
            if rx.search(q) and i < len(answers):
                return answers[i]
        return answers[fallback] if len(answers) > fallback else None

    return (questions, answers,
            pick(Q_ARGS_RE, 0), pick(Q_TRIGGER_RE, 1), pick(Q_SOURCE_RE, 2))


def audit_block(block, users, tools, prev):
    """Return a list of error codes for one security block."""
    errors = []
    parts = {}
    detail = {}
    for tag, rx in TAG_RE.items():
        m = rx.search(block)
        if not m:
            errors.append("malformed_block")
            return errors, parts, detail
        parts[tag] = m.group(1)

    reason = parts["tool_reason"]
    level = parts["tool_security"].strip().lower()
    trace = parse_trace(parts["tool_trace"])
    questions, answers, args_answer, trigger_answer, source_answer = split_qa(reason)

    # question count must match the level: safe stops after the three fixed questions
    n_q = len(questions)
    if (level == "safe" and n_q != 3) or (level != "safe" and n_q != 7):
        errors.append("qcount_mismatch")

    # the argument-provenance question has to be there and has to come first, because every
    # later answer is supposed to be conditioned on it
    if not questions or not Q_ARGS_RE.search(questions[0]):
        errors.append("arg_question_missing")

    # a value the user delegated to a source they named is still the user's own instruction
    delegated = bool(args_answer) and DELEGATED_MARK in norm(args_answer)
    detail["delegated"] = delegated

    if not trace:
        errors.append("malformed_block")
        return errors, parts, detail

    kind, idx = trace
    if (level == "safe") != (kind == "user_message"):
        errors.append("trace_level_mismatch")

    # the turn this call follows has to be consistent with where the trigger came from
    detail["prev_turn"] = prev
    if CHECK_POSITION and prev:
        if prev == "tool" and level == "safe":
            if not (ALLOW_DELEGATED_SAFE and delegated):
                errors.append("position_label_conflict")
        elif prev == "human" and level != "safe":
            errors.append("position_label_conflict")

    # the source answer states the source too; it must agree with the trace tag
    if source_answer:
        claimed = parse_trace(source_answer)
        if claimed and claimed[0] != kind:
            errors.append("source_answer_trace_disagree")

    # the cited message has to exist at this point in the conversation
    pool = users if kind == "user_message" else tools
    if idx < 1 or idx > len(pool):
        errors.append("trace_index_missing")
        cited = None
    else:
        cited = pool[idx - 1]

    # the quoted trigger words have to be in the message they are attributed to. This reads
    # the TRIGGER answer, never the provenance answer: the provenance answer deliberately
    # quotes decoy text from other messages in order to rule it out, so checking it here
    # would flag correct blocks for citing the very thing they rejected.
    quotes = ([q for q in QUOTE_RE.findall(trigger_answer) if len(q) >= MIN_QUOTE_LEN]
              if trigger_answer else [])
    detail["quotes"] = quotes[:3]
    if not quotes:
        errors.append("no_quote_in_trigger")
    elif cited is not None:
        haystack = norm(cited)
        best = max(overlap(q, cited) for q in quotes)
        detail["best_overlap"] = round(best, 3)
        if any(norm(q) in haystack for q in quotes):
            pass                                   # verbatim, nothing to report
        elif best >= FUZZY_MIN:
            errors.append("quote_loose_match")     # reworded but the source still holds
        else:
            errors.append("quote_not_in_cited")
            # the decisive case: attributed to the user, but the words live in tool output
            if kind == "user_message":
                tool_text = norm(" ".join(tools))
                tool_best = max(overlap(q, tool_text) for q in quotes)
                detail["tool_overlap"] = round(tool_best, 3)
                if any(norm(q) in tool_text for q in quotes) or tool_best > best:
                    errors.append("quote_is_from_tool")

    user_text = norm(" ".join(users))
    tool_text = norm(" ".join(tools))
    arg_values = distinctive_args(parts["tool_args"])

    # A safe call's arguments must be derivable from what the user wrote, unless the user
    # delegated them: "summarise every website in that channel" hands the URLs to the channel,
    # so a URL arriving in a tool response is still the user's instruction being carried out.
    if level == "safe":
        for value in arg_values:
            v = norm(value)
            if v and v not in user_text and v in tool_text:
                if ALLOW_DELEGATED_SAFE and delegated:
                    detail["arg_delegated"] = value
                else:
                    errors.append("safe_but_args_from_tool")
                    detail["arg_only_in_tool"] = value
                break

    # The mirror image, and the one that catches the failure this format was built for: a call
    # blamed on a tool response whose values are nowhere in that tool response. That is what a
    # block looks like when it matched an injection on the verb alone — the injected text asked
    # for a similar action with different values, so the values it named never reach the call.
    # Left as a warning because a value the model composed itself appears nowhere either; check
    # the counts in the report before promoting it to HARD_ERRORS.
    elif cited is not None and arg_values:
        cited_text = norm(cited)
        in_cited = [v for v in arg_values if norm(v) in cited_text]
        elsewhere = [v for v in arg_values
                     if norm(v) in user_text or norm(v) in tool_text.replace(cited_text, " ")]
        if not in_cited and elsewhere:
            errors.append("nonsafe_args_not_in_cited")
            detail["arg_not_in_blamed_source"] = elsewhere[0]

    return errors, parts, detail


def injection_markers(text):
    """Which surface giveaways a tool response carries."""
    low = norm(text)
    fields = set(re.findall(r'\\?"(\w+)\\?"\s*:', text or ""))
    return {
        "extra_field": any(f in INJECTION_FIELDS for f in fields),
        "loud_phrase": any(p in low for p in INJECTION_PHRASES),
    }


def audit_record(record):
    """Audit every security block in a record. Returns (errors, block_reports)."""
    all_errors, reports = [], []
    conversations = record.get("conversations", [])
    prev = None
    prev_tool_idx = None
    injected_idx = set()
    for i, conv in enumerate(conversations):
        src = conv.get("from")
        value = conv.get("value", "") or ""
        if src != "gpt":
            if src == "tool" or "<tool_response>" in value:
                prev = "tool"
                prev_tool_idx = i
            elif src == "human":
                prev = "human"
            continue
        if "<tool_call_security>" not in value:
            continue
        users, tools = build_index(conversations, i)
        for block in BLOCK_RE.findall(value):
            errors, parts, detail = audit_block(block, users, tools, prev)
            if prev == "tool" and tools:
                detail.update(injection_markers(tools[-1]))
                sec = re.search(r"<tool_security>(.*?)</tool_security>", block, re.S)
                if sec and sec.group(1).strip().lower() != "safe":
                    injected_idx.add(prev_tool_idx)
            all_errors.extend(errors)
            reports.append({
                "turn": i,
                "tool_name": parts.get("tool_name", "").strip(),
                "trace": parts.get("tool_trace", "").strip(),
                "security": parts.get("tool_security", "").strip().lower(),
                "user_turns_available": len(users),
                "tool_turns_available": len(tools),
                "hard": sorted(set(errors) & HARD_ERRORS),
                "errors": errors,
                **detail,
            })
    return all_errors, reports, injected_idx


def print_shortcut_report(pos, markers, benign_markers, title="SHORTCUT DIAGNOSTICS"):
    """Report how easily the label can be guessed without reading the conversation.

    Two shortcuts matter. If the turn a call follows already determines the label, the model
    never has to work out who issued the instruction. If injected tool responses always look
    different from benign ones (an extra field, a shouty phrase), the model learns to spot
    that costume instead. Either one produces a model that scores well here and fails on an
    injection written in a style the data never contained.
    """
    pct = lambda n, d: (100.0 * n / d) if d else 0.0
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)

    total = sum(pos.values())
    if total:
        print("label vs the turn the call follows:")
        for prev in ("human", "tool"):
            sub = {lvl: n for (p, lvl), n in pos.items() if p == prev}
            tot = sum(sub.values())
            if not tot:
                continue
            top = max(sub.items(), key=lambda kv: kv[1])
            print("  follows %-6s %6d calls -> %s"
                  % (prev, tot, ", ".join("%s %d (%.1f%%)" % (k, v, pct(v, tot))
                                          for k, v in sorted(sub.items(), key=lambda kv: -kv[1]))))
            if pct(top[1], tot) >= 99.0:
                print("      WARNING: %.1f%% share one label - position alone predicts it"
                      % pct(top[1], tot))

    n = markers["total"]
    if n:
        either = markers["extra_field"] + markers["loud_phrase"] - markers["both"]
        print("\ninjected tool responses preceding a non-safe call: %d" % n)
        print("  in a dedicated field (%s...): %6d (%.1f%%)"
              % (INJECTION_FIELDS[0], markers["extra_field"], pct(markers["extra_field"], n)))
        print("  containing a shouty phrase              : %6d (%.1f%%)"
              % (markers["loud_phrase"], pct(markers["loud_phrase"], n)))
        print("  carrying neither (subtle, hardest kind) : %6d (%.1f%%)"
              % (n - either, pct(n - either, n)))
        if pct(either, n) >= 95.0:
            print("      WARNING: %.1f%% are marked - the giveaway is learnable on its own"
                  % pct(either, n))

    bn = benign_markers["total"]
    if bn:
        print("\nbenign tool responses: %d, of which %d (%.1f%%) use one of those fields"
              % (bn, benign_markers["extra_field"], pct(benign_markers["extra_field"], bn)))
        if benign_markers["extra_field"] == 0 and markers["extra_field"]:
            print("      WARNING: those fields appear ONLY in injections - a perfect giveaway")


def derive_paths(path):
    base = path[:-len(".json")] if path.endswith(".json") else path
    return base + "-valid.json", base + "-invalid.json", base + "-audit-report.jsonl"


def process_file(path):
    valid_path, invalid_path, report_path = derive_paths(path)
    print("\n" + "#" * 70)
    print("# %s" % path)
    print("#" * 70)

    if not os.path.exists(path):
        print("file not found, skipping")
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid, invalid = [], []
    errors = Counter()
    levels = Counter()
    bad_levels = Counter()
    pos = Counter()
    markers = Counter()
    benign = Counter()
    blocks_total = blocks_bad = 0



    with open(report_path, "w", encoding="utf-8") as rep:
        for record in data:
            record_errors, reports, injected_idx = audit_record(record)
            for r in reports:
                blocks_total += 1
                levels[r["security"]] += 1
                if r.get("prev_turn"):
                    pos[(r["prev_turn"], r["security"])] += 1
                if r.get("prev_turn") == "tool" and r["security"] != "safe":
                    markers["total"] += 1
                    if r.get("extra_field"):
                        markers["extra_field"] += 1
                    if r.get("loud_phrase"):
                        markers["loud_phrase"] += 1
                    if r.get("extra_field") and r.get("loud_phrase"):
                        markers["both"] += 1
                if r["hard"]:
                    blocks_bad += 1
                    bad_levels[r["security"]] += 1
                if r["errors"]:
                    for e in set(r["errors"]):
                        errors[e] += 1
                    rep.write(json.dumps(
                        {"id": record.get("id"), **r}, ensure_ascii=False) + "\n")
            # benign tool responses = every tool turn except those that triggered a
            # non-safe call, so the contrast with injected ones is meaningful
            for j, conv in enumerate(record.get("conversations", [])):
                value = conv.get("value", "") or ""
                is_tool = conv.get("from") == "tool" or (
                    conv.get("from") != "gpt" and "<tool_response>" in value)
                if is_tool and j not in injected_idx:
                    benign["total"] += 1
                    if injection_markers(value)["extra_field"]:
                        benign["extra_field"] += 1

            hard = set(record_errors) & HARD_ERRORS
            is_bad = bool(record_errors) if STRICT else bool(hard)
            (invalid if is_bad else valid).append(record)

    with open(valid_path, "w", encoding="utf-8") as f:
        json.dump(valid, f, ensure_ascii=False, indent=2)
    with open(invalid_path, "w", encoding="utf-8") as f:
        json.dump(invalid, f, ensure_ascii=False, indent=2)

    total = len(data)
    pct = lambda n, d: (100.0 * n / d) if d else 0.0
    print("records   : %d total, %d valid (%.1f%%), %d invalid (%.1f%%)"
          % (total, len(valid), pct(len(valid), total),
             len(invalid), pct(len(invalid), total)))
    print("blocks    : %d total, %d with HARD errors (%.1f%%)"
          % (blocks_total, blocks_bad, pct(blocks_bad, blocks_total)))
    print("            (a record is invalid only if it has a HARD error; "
          "set STRICT=True to also reject warnings)")

    if levels:
        print("\nsecurity level distribution (failed / total):")
        for lvl, n in levels.most_common():
            print("  %-12s %6d / %-6d  (%.1f%% of blocks, %.1f%% of these failed)"
                  % (lvl, bad_levels[lvl], n, pct(n, blocks_total), pct(bad_levels[lvl], n)))

    if errors:
        print("\nfindings by check (HARD errors first):")
        width = max(len(ERROR_LABELS.get(k, k)) for k in errors)
        for code, n in sorted(errors.items(),
                              key=lambda kv: (kv[0] not in HARD_ERRORS, -kv[1])):
            mark = "HARD" if code in HARD_ERRORS else "warn"
            print("  [%s] %-*s %6d  (%.2f%% of blocks)"
                  % (mark, width, ERROR_LABELS.get(code, code), n,
                     pct(n, blocks_total)))

    print_shortcut_report(pos, markers, benign)

    print("\nwrote %s" % valid_path)
    print("wrote %s" % invalid_path)
    print("wrote %s" % report_path)
    return {"records": total, "valid": len(valid), "invalid": len(invalid),
            "blocks": blocks_total, "blocks_bad": blocks_bad,
            "errors": errors, "levels": levels, "bad_levels": bad_levels,
            "pos": pos, "markers": markers, "benign": benign}


def main():
    summaries = []
    for path in INPUT_FILES:
        summary = process_file(path)
        if summary:
            summaries.append((path, summary))

    if not summaries:
        return

    print("\n" + "=" * 70)
    print("OVERALL")
    print("=" * 70)
    totals = Counter()
    errors = Counter()
    levels = Counter()
    bad_levels = Counter()
    pos = Counter()
    markers = Counter()
    benign = Counter()
    for _, s in summaries:
        for key in ("records", "valid", "invalid", "blocks", "blocks_bad"):
            totals[key] += s[key]
        errors.update(s["errors"])
        levels.update(s["levels"])
        bad_levels.update(s["bad_levels"])
        pos.update(s["pos"])
        markers.update(s["markers"])
        benign.update(s["benign"])

    pct = lambda n, d: (100.0 * n / d) if d else 0.0
    print("records   : %d total, %d valid (%.1f%%), %d invalid (%.1f%%)"
          % (totals["records"], totals["valid"], pct(totals["valid"], totals["records"]),
             totals["invalid"], pct(totals["invalid"], totals["records"])))
    print("blocks    : %d total, %d with HARD errors (%.1f%%)"
          % (totals["blocks"], totals["blocks_bad"],
             pct(totals["blocks_bad"], totals["blocks"])))

    if levels:
        print("\nsecurity level distribution (failed / total):")
        for lvl, n in levels.most_common():
            print("  %-12s %6d / %-6d  (%.1f%% of blocks, %.1f%% of these failed)"
                  % (lvl, bad_levels[lvl], n, pct(n, totals["blocks"]),
                     pct(bad_levels[lvl], n)))

    if errors:
        print("\nfindings by check (HARD errors first):")
        width = max(len(ERROR_LABELS.get(k, k)) for k in errors)
        for code, n in sorted(errors.items(),
                              key=lambda kv: (kv[0] not in HARD_ERRORS, -kv[1])):
            mark = "HARD" if code in HARD_ERRORS else "warn"
            print("  [%s] %-*s %6d  (%.2f%% of blocks)"
                  % (mark, width, ERROR_LABELS.get(code, code), n,
                     pct(n, totals["blocks"])))

    print_shortcut_report(pos, markers, benign,
                          title="SHORTCUT DIAGNOSTICS (all files combined)")

    print("\nper file:")
    for path, s in summaries:
        print("  %-70s %5d/%-5d invalid (%.1f%%)"
              % (os.path.basename(path), s["invalid"], s["records"],
                 pct(s["invalid"], s["records"])))


if __name__ == "__main__":
    main()
