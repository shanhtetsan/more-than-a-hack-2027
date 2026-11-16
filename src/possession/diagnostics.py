"""The end-of-run diagnostic summary: counts only, never per-frame data.

Built from the stats core.run_match returns, plus the restart tags (frame -> Type) so each tag
type can be reported as parsed / inside a graded period / used to end a D.
"""

from collections import Counter

CAUSES = ("period start", "ball out", "possible goal")
ENDINGS = ("restart tag", "control exit", "goal restart tag", "never ended")


def restart_tag_counts(types, used, blocks):
    """{type: {"parsed", "in graded", "ended a D"}} for restart tags keyed by anchor frame."""
    def graded(frame):
        return any(first <= frame <= last for _, first, last in blocks)

    out = {}
    for frame, kind in types.items():
        row = out.setdefault(kind or "(no type)", Counter())
        row["parsed"] += 1
        row["in graded"] += graded(frame)
        row["ended a D"] += frame in used
    return out


def summary_lines(stats, types, blocks):
    """The summary as a list of printable lines."""
    lines = ["", "=== diagnostics (counts only) ==="]

    letters, graded = stats["letters"], stats["graded_letters"]
    lines.append("labels, all frames:    " + "  ".join(f"{k} {letters[k]}" for k in "HADX"))
    lines.append("labels, graded frames: " + "  ".join(f"{k} {graded[k]}" for k in "HADX"))

    lines.append("D segments before merging, by cause -> how they ended:")
    segments = stats["d_segments"]
    for cause in CAUSES:
        row = {how: segments.get((cause, how), 0) for how in ENDINGS}
        lines.append(f"  {cause:<14} total {sum(row.values()):>4}   "
                     + "  ".join(f"{how} {n}" for how, n in row.items()))
    for half, frames in zip((b[0] for b in blocks), stats["period_start_frames"], strict=False):
        lines.append(f"  period {half}: {frames} frames of D before its first restart")

    lines.append("restart tags by type: parsed / anchor inside a graded period / ended a D")
    tags = restart_tag_counts(types, stats["used_restarts"], blocks)
    for kind in sorted(tags):
        row = tags[kind]
        lines.append(f"  {kind:<24} {row['parsed']:>4} {row['in graded']:>4} "
                     f"{row['ended a D']:>4}")
    total = Counter()
    for row in tags.values():
        total.update(row)
    lines.append(f"  {'total':<24} {total['parsed']:>4} {total['in graded']:>4} "
                 f"{total['ended a D']:>4}")

    phases = stats.get("phases")
    if phases is not None:
        lines.append("Smart Tagging phases, checked against our own confirmed-control frames:")
        lines.append("  code: anchor frames / agree with tagged team -> use "
                     "(+1 team has the ball, -1 team is without it, 0 unused)")
        for code, (agree, total, use) in phases["codes"].items():
            share = f"{agree / total:.2f}" if total else "n/a"
            lines.append(f"  {code:<30} {total:>6} {share:>5} -> {use:+d}")
        lines.append(f"  frames with a phase vote: {phases['vote frames']}; confirmed starts "
                     f"added by phases: {phases['runs']}")
    stop = stats.get("stoppage")
    if stop is not None:
        lines.append(f"stoppages from collective speed: {stop['runs']} runs, {stop['frames']} "
                     f"graded frames; speed known on {stop['speed known']} graded frames, "
                     f"quartiles (m/s) {stop['speed quartiles']}")
    runs = stats["confirmed_runs"]
    lines.append(f"confirmed-control runs in graded frames: H {runs['H']}  A {runs['A']}")
    final = stats["final"]
    lines.append(f"possession switches by confirmed control (state machine): "
                 f"{stats['control_switches']}")
    lines.append(f"final labels: possessions {final.get('possessions', 0)}  "
                 f"team changes {final.get('team changes', 0)}  "
                 f"direct H<->A {final.get('direct H<->A', 0)}  "
                 f"D segments {final.get('D segments', 0)}")

    lines.append("ball position: observed / interpolated / missing")
    for scope, row in stats["ball"].items():
        lines.append(f"  {scope:<14} {row['observed']:>7} {row['interpolated']:>7} "
                     f"{row['missing']:>7}")
    return lines
