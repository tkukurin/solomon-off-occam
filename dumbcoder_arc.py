"""ARC-AGI-1 program induction with the dumbcoder enumerator.

Single-task demo: best-first enumeration over a small slice of Hodel's
arc-dsl (refs/arc-dsl) finds every minimal program reproducing the train
pairs of one ARC-AGI-1 task (Chollet, 2019), then scores the induced
programs on the held-out test pair.

The DSL is grid-typed only -- mirrors, rotations, halves, and
concatenation -- so search is pure typed enumeration ordered by program
size, the uniform-prior special case of dumbcoder's ECD explore phase.
"""

import heapq
import json
import sys
import urllib.request
from dataclasses import replace
from pathlib import Path

import dumbcoder as dc

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "refs" / "arc-dsl"))
import dsl as arc  # noqa: E402

GRID = "grid"
INPUT = dc.Delta("$I", type=GRID, isarg=True)
TASK_ID = "4c4377d9"
TASK_PATH = ROOT / "data" / "arc" / f"{TASK_ID}.json"
TASK_URL = ("https://raw.githubusercontent.com/fchollet/ARC-AGI/"
            f"master/data/training/{TASK_ID}.json")
WANTED = 2
BUDGET = 1_000_000


def primitives() -> list[dc.Delta]:
    # a generic grid-manipulation vocabulary, not tailored to the task:
    # the solution needs 3 of these 13, the rest are honest distractors
    unary = (arc.hmirror, arc.vmirror, arc.dmirror, arc.cmirror,
             arc.rot90, arc.rot180, arc.rot270,
             arc.tophalf, arc.bottomhalf, arc.lefthalf, arc.righthalf)
    binary = (arc.hconcat, arc.vconcat)
    return [dc.Delta(head=f, type=GRID, tailtypes=(GRID,) * arity, repr_str=f.__name__)
            for arity, fs in ((1, unary), (2, binary)) for f in fs]


def load_task() -> dict:
    if not TASK_PATH.exists():
        TASK_PATH.parent.mkdir(parents=True, exist_ok=True)
        TASK_PATH.write_bytes(urllib.request.urlopen(TASK_URL).read())
    raw = json.loads(TASK_PATH.read_text())
    grid = lambda g: tuple(tuple(row) for row in g)
    return {split: tuple((grid(p["input"]), grid(p["output"])) for p in raw[split])
            for split in ("train", "test")}


def run_program(tree: dc.Delta, grid: tuple) -> tuple | None:
    bound = dc.replace_hidden(tree, INPUT, dc.Delta(grid, type=GRID))
    try:
        return bound()
    except Exception:  # boundary: synthesized programs may raise anything
        return None


def solve(pairs, dsl: dc.Deltas, wanted=WANTED, budget=BUDGET, log=lambda _: None):
    """Best-first enumeration with unit costs == smallest programs first;
    collects the first `wanted` programs consistent with all train pairs."""
    frontier = [(0.0, 0, dc.Delta("<>", type=GRID, ishole=True))]
    pushed, expanded, tested = 1, 0, 0
    programs = []

    while frontier and expanded < budget and len(programs) < wanted:
        cost, _, tree = heapq.heappop(frontier)

        if (path := dc._find_hole(tree)) is None:
            tested += 1
            if all(run_program(tree, i) == o for i, o in pairs):
                programs.append(tree)
                log(f"  hit #{len(programs)}: {tree} "
                    f"({dc.length(tree)} nodes, {tested} tested, {expanded} expanded)")
            continue

        hole = tree
        for index in path:
            hole = hole.tails[index]
        for primitive_index in dsl.bytype[hole.type]:
            primitive = dsl[primitive_index]
            tails = tuple(dc.Delta("<>", ishole=True, type=t)
                          for t in primitive.tailtypes)
            expanded_tree = replace(primitive, tails=tails) if tails else primitive
            heapq.heappush(
                frontier, (cost + 1.0, pushed, dc._replace_at_path(tree, path, expanded_tree)))
            pushed += 1
            expanded += 1

    return programs, tested, expanded


def render(grid: tuple | None) -> list[str]:
    if grid is None:
        return ["<error>"]
    return [" ".join(str(v) if v else "." for v in row) for row in grid]


def show_pair(tag: str, given: tuple, predicted: tuple | None, expected: tuple):
    verdict = "MATCH" if predicted == expected else "MISMATCH"
    blocks = (render(given), render(predicted), render(expected))
    width = max(len(line) for block in blocks for line in block)
    print(f"\n{tag}: input -> predicted | expected  [{verdict}]")
    for row in range(max(map(len, blocks))):
        cells = (block[row] if row < len(block) else "" for block in blocks)
        print("  " + " | ".join(cell.ljust(width) for cell in cells))


def main() -> None:
    task = load_task()
    dsl = dc.Deltas([*primitives(), INPUT])
    print(f"== ARC-AGI-1 task {TASK_ID}: {len(task['train'])} train pairs, "
          f"{len(dsl)} primitives ==")

    programs, tested, expanded = solve(task["train"], dsl, log=print)
    print(f"search: {tested} complete programs tested, {expanded} expansions")
    if not programs:
        raise ValueError(f"no program found within {BUDGET} expansions")

    print("\n== induced programs on held-out test ==")
    for tree in programs:
        correct = sum(run_program(tree, i) == o for i, o in task["test"])
        print(f"  {tree}  --  {correct}/{len(task['test'])} test pairs")

    best = programs[0]
    for index, (given, expected) in enumerate(task["test"]):
        show_pair(f"test[{index}] via {best}", given, run_program(best, given), expected)


if __name__ == "__main__":
    main()
