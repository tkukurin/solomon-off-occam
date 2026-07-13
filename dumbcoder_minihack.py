"""DreamCoder-style parameterized skill learning in real MiniHack rooms.

Run ``uv run --extra minihack dumbcoder_minihack.py`` to open the dynamic demo.
The website edits the starting DSL, priors, compression settings, and task cases,
then reruns real MiniHack training and held-out evaluation directly.
"""

import base64
import heapq
import json
import threading
import webbrowser
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from enum import Enum
from io import BytesIO
from math import log
from pathlib import Path
from typing import Any, Callable, cast

import minihack  # noqa: F401 -- import registers environments and loads NLE
import numpy as np
from PIL import Image
from simple_parsing import parse

import dumbcoder as dc

PREDICATE = "state_predicate"
POLICY = "reactive_policy"
DEFAULT_CONFIG_PATH = Path("minihack_config.json")
RESULT_PATH = Path("experiments/2607/13-minihack-program-learning-results.json")
FRAME_PATH = Path("experiments/2607/13-minihack-program-learning-trajectory.txt")
HTML_PATH = Path("experiments/2607/13-minihack-program-learning-demo.html")
ACTION_NAMES = ("north", "east", "south", "west", "wait")
ACTION_DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0), (0, 0))
MINIHACK_ACTIONS = (
    *(getattr(nethack.CompassDirection, name) for name in ("N", "E", "S", "W")),
    nethack.MiscDirection.WAIT,
)


@dataclass(frozen=True)
class State:
    player_x: int
    player_y: int
    goal_x: int
    goal_y: int


@dataclass
class Rollout:
    success: bool
    total_reward: float
    rgb_frames: list[str]
    terminal_frame_inferred: bool = False


@dataclass
class SearchResult:
    program: dc.Delta
    rollouts: list[Rollout]
    programs_tested: int
    nodes_expanded: int


@dataclass
class LibraryPrimitive:
    delta: dc.Delta
    kind: str
    weight: float
    definition: str | None = None
    learned_at: int | None = None


@dataclass(frozen=True)
class PatternVariable:
    index: int
    output_type: str


@dataclass(frozen=True)
class PatternNode:
    symbol: str
    output_type: str
    children: tuple[Pattern, ...] = ()


Pattern = PatternVariable | PatternNode


@dataclass
class SkillCandidate:
    pattern: PatternNode
    support: int
    fixed_nodes: int
    arity: int
    template_gain: int
    strict_ast_gain: int
    bindings: list[dict[int, dc.Delta]]


class Mode(Enum):
    serve = "serve"
    run = "run"


@dataclass
class Cli:
    """MiniHack program-learning demo."""

    mode: Mode = Mode.serve
    config: Path = DEFAULT_CONFIG_PATH
    host: str = "127.0.0.1"
    port: int = 8000
    open_browser: bool = True
    render: bool = False


def constant_policy(action: int) -> Callable[[State], int]:
    return lambda _state: action


def branch(
    predicate: Callable[[State], bool],
    when_true: Callable[[State], int],
    when_false: Callable[[State], int],
) -> Callable[[State], int]:
    return lambda state: when_true(state) if predicate(state) else when_false(state)


def primitive_from_spec(spec: dict[str, Any]) -> dc.Delta:
    name = str(spec["name"])
    kind = str(spec["kind"])
    if kind == "action":
        if name not in ACTION_NAMES:
            raise ValueError(f"unknown action primitive: {name}")
        return dc.Delta(
            constant_policy(ACTION_NAMES.index(name)), type=POLICY, repr_str=name
        )
    if kind == "predicate":
        predicates: dict[str, Callable[[State], bool]] = {
            "goal_east": lambda state: state.player_x < state.goal_x,
            "goal_west": lambda state: state.player_x > state.goal_x,
            "goal_south": lambda state: state.player_y < state.goal_y,
            "goal_north": lambda state: state.player_y > state.goal_y,
        }
        if name not in predicates:
            raise ValueError(f"unknown predicate primitive: {name}")
        return dc.Delta(predicates[name], type=PREDICATE, repr_str=name)
    if kind == "branch" and name == "if":
        return dc.Delta(
            branch,
            type=POLICY,
            tailtypes=(PREDICATE, POLICY, POLICY),
            repr_str="if",
        )
    raise ValueError(f"unknown primitive kind/name: {kind}/{name}")


def load_config(path: Path) -> dict[str, Any]:
    config = cast(dict[str, Any], json.loads(path.read_text()))
    if config.get("prior_mode") not in {"uniform", "weighted"}:
        raise ValueError("prior_mode must be 'uniform' or 'weighted'")
    if not config.get("train_tasks"):
        raise ValueError("train_tasks must contain at least two tasks")
    if not config.get("heldout_tasks"):
        raise ValueError("heldout_tasks must contain at least one task")
    if not config.get("library"):
        raise ValueError("library must contain at least one primitive")
    return config


def initial_library(config: dict[str, Any]) -> list[LibraryPrimitive]:
    library: list[LibraryPrimitive] = []
    for spec in config["library"]:
        if not spec.get("enabled", True):
            continue
        weight = float(spec.get("weight", 1.0))
        if weight <= 0:
            raise ValueError(f"primitive {spec['name']} must have positive weight")
        library.append(
            LibraryPrimitive(
                delta=primitive_from_spec(spec),
                kind=str(spec["kind"]),
                weight=weight,
            )
        )
    if not library:
        raise ValueError("at least one library primitive must be enabled")
    return library


def clone_library(library: list[LibraryPrimitive]) -> list[LibraryPrimitive]:
    return [replace(primitive) for primitive in library]


def symbol(tree: dc.Delta) -> str:
    return str(tree.repr_str or tree.head)


def tree_key(tree: dc.Delta) -> tuple[Any, ...]:
    return (symbol(tree), str(tree.type), tuple(tree_key(tail) for tail in tree.tails))


def iter_program_nodes(tree: dc.Delta):
    yield tree
    for tail in tree.tails:
        yield from iter_program_nodes(tail)


def production_probabilities(
    library: list[LibraryPrimitive], prior_mode: str
) -> list[float]:
    weights = [1.0 if prior_mode == "uniform" else p.weight for p in library]
    totals: defaultdict[str, float] = defaultdict(float)
    for primitive, weight in zip(library, weights):
        totals[str(primitive.delta.type)] += weight
    return [
        weight / totals[str(primitive.delta.type)]
        for primitive, weight in zip(library, weights)
    ]


def library_snapshot(
    library: list[LibraryPrimitive], prior_mode: str
) -> list[dict[str, Any]]:
    probabilities = production_probabilities(library, prior_mode)
    return [
        {
            "name": symbol(primitive.delta),
            "kind": primitive.kind,
            "output_type": str(primitive.delta.type),
            "inputs": list(primitive.delta.tailtypes),
            "weight": primitive.weight,
            "probability": probability,
            "definition": primitive.definition,
            "learned_at": primitive.learned_at,
        }
        for primitive, probability in zip(library, probabilities)
    ]


def state_from_observation(observation: dict[str, np.ndarray]) -> State:
    goal_y, goal_x = np.where(observation["chars"] == ord(">"))
    if len(goal_x) != 1:
        raise ValueError(f"expected one visible staircase, found {len(goal_x)}")
    player_x, player_y = observation["blstats"][:2]
    return State(int(player_x), int(player_y), int(goal_x[0]), int(goal_y[0]))
def room_bounds(
    observation: dict[str, np.ndarray],
) -> tuple[int, int, int, int] | None:
    chars = observation["chars"]
    visible = np.isin(chars, np.fromiter(map(ord, ".@><|-"), dtype=np.uint8))
    rows, columns = np.where(visible)
    if not len(rows):
        return None
    return int(rows.min()), int(rows.max()), int(columns.min()), int(columns.max())


def ascii_frame(observation: dict[str, np.ndarray]) -> str | None:
    bounds = room_bounds(observation)
    if bounds is None:
        return None
    top, bottom, left, right = bounds
    crop = observation["chars"][top:bottom + 1, left:right + 1]
    return "\n".join(
        "".join(chr(int(value)) if 32 <= value < 127 else " " for value in row)
        for row in crop
    )


def rgb_frame(observation: dict[str, np.ndarray]) -> str | None:
    bounds = room_bounds(observation)
    if bounds is None or "pixel" not in observation:
        return None
    top, bottom, left, right = bounds
    pixel = observation["pixel"]
    tile_height = pixel.shape[0] // observation["chars"].shape[0]
    tile_width = pixel.shape[1] // observation["chars"].shape[1]
    crop = pixel[
        top * tile_height:(bottom + 1) * tile_height,
        left * tile_width:(right + 1) * tile_width,
    ]
    buffer = BytesIO()
    Image.fromarray(crop).save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def advance_ascii_frame(frame: str, action: int) -> str:
    rows = [list(row) for row in frame.splitlines()]
    player_y, player_x = next(
        (y, x)
        for y, row in enumerate(rows)
        for x, value in enumerate(row)
        if value == "@"
    )
    dx, dy = ACTION_DELTAS[action]
    rows[player_y][player_x] = "."
    rows[player_y + dy][player_x + dx] = "@"
    return "\n".join("".join(row) for row in rows)

    size = case["size"]
    width, height = (
        (int(size), int(size))
        if isinstance(size, int)
        else (int(size[0]), int(size[1]))
    )
    start = tuple(map(int, case["start"]))
    goal = tuple(map(int, case["goal"]))
    level = LevelGenerator(w=width, h=height, lit=True)
    if "corridor_row" in case:
        corridor_row = int(case["corridor_row"])
        for y in range(height):
            if y == corridor_row:
                continue
            for x in range(width):
                level.add_terrain((x, y), " ")
    for wall in case.get("walls", []):
        level.add_terrain(tuple(map(int, wall)), "|")
    level.set_start_pos(start)
    level.add_goal_pos(goal)
    return MiniHackNavigation(
        des_file=level.get_des(),
        actions=MINIHACK_ACTIONS,
        observation_keys=("blstats", "chars", "pixel"),
        max_episode_steps=int(case["max_steps"]),
    )


def run_policy(
    environment: MiniHackNavigation,
    policy: Callable[[State], int],
    max_steps: int,
    seed: int,
) -> Rollout:
    observation, _ = environment.reset(seed=seed)
    total_reward = 0.0
    trace: list[dict[str, Any]] = []
    initial_frame = ascii_frame(observation)
    if initial_frame is None:
    initial_rgb = rgb_frame(observation)
    rgb_frames = [initial_rgb] if initial_rgb is not None else []
    terminal_frame_inferred = False

    for step_index in range(max_steps):
        state = state_from_observation(observation)
            return Rollout(False, total_reward, trace, frames, rgb_frames)
        observation, reward, terminated, truncated, _ = environment.step(action)
        total_reward += float(reward)
        next_frame = ascii_frame(observation)
        if next_frame is None:
            if terminated and reward > 0:
                next_frame = advance_ascii_frame(frames[-1], action)
                terminal_frame_inferred = True
            else:
        next_rgb = rgb_frame(observation)
        if rgb_frames:
            rgb_frames.append(next_rgb if next_rgb is not None else rgb_frames[-1])
        trace.append({
            "step": step_index,
            "player": [state.player_x, state.player_y],
            "goal": [state.goal_x, state.goal_y],
            "action": ACTION_NAMES[action],
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        })
        if terminated or truncated:
            return Rollout(
                bool(terminated and reward > 0),
                total_reward,
                rgb_frames,
                terminal_frame_inferred,
            )
    return Rollout(
        False, total_reward, trace, frames, rgb_frames, terminal_frame_inferred
    )


def find_hole(tree: dc.Delta) -> list[int] | None:
    if tree.ishole:
        return []
    for index, tail in enumerate(tree.tails):
        path = find_hole(tail)
        if path is not None:
            return [index, *path]
    return None


def replace_at_path(tree: dc.Delta, path: list[int], node: dc.Delta) -> dc.Delta:
    if not path:
        return node
    index = path[0]
    tails = list(tree.tails)
    tails[index] = replace_at_path(tails[index], path[1:], node)
    return replace(tree, tails=tuple(tails))


def expand_hole(
    tree: dc.Delta,
    path: list[int],
    dsl: dc.Deltas,
    costs: list[float],
    cost: float,
    sequence: int,
    frontier: list[tuple[float, int, dc.Delta]],
    max_depth: int,
) -> tuple[int, int]:
    hole = tree
    for index in path:
        hole = hole.tails[index]
    candidates = (
        dsl.bytype_terminal.get(hole.type, ())
        if len(path) >= max_depth
        else dsl.bytype.get(hole.type, ())
    )
    expanded = 0
    for primitive_index in candidates:
        primitive = dsl[primitive_index]
        tails = tuple(
            dc.Delta(f"<>:{tail_type}", type=tail_type, ishole=True)
            for tail_type in primitive.tailtypes
        )
        node = replace(primitive, tails=tails) if tails else primitive
        sequence += 1
        expanded += 1
        heapq.heappush(
            frontier,
            (cost + costs[primitive_index], sequence, replace_at_path(tree, path, node)),
        )
    return sequence, expanded


def evaluate_program(
    environments: list[MiniHackNavigation],
    cases: list[dict[str, Any]],
    program: dc.Delta,
    seed: int,
) -> list[Rollout] | None:
    policy = cast(Callable[[State], int], program())
    rollouts = []
    for environment, case in zip(environments, cases):
        rollout = run_policy(environment, policy, int(case["max_steps"]), seed)
        if not rollout.success:
            return None
        rollouts.append(rollout)
    return rollouts


def synthesize(
    environments: list[MiniHackNavigation],
    task: dict[str, Any],
    library: list[LibraryPrimitive],
    config: dict[str, Any],
    prior_mode: str,
) -> SearchResult:
    dsl = dc.Deltas([primitive.delta for primitive in library])
    probabilities = production_probabilities(library, prior_mode)
    costs = [-log(probability) for probability in probabilities]
    root = dc.Delta(f"<>:{POLICY}", type=POLICY, ishole=True)
    frontier = [(0.0, 0, root)]
    sequence = 0
    programs_tested = 0
    nodes_expanded = 0
    budget = int(config["search_budget"])

    while frontier and nodes_expanded < budget:
        cost, _, tree = heapq.heappop(frontier)
        path = find_hole(tree)
        if path is None:
            programs_tested += 1
            try:
                rollouts = evaluate_program(
                    environments,
                    task["cases"],
                    tree,
                    int(config["seed"]),
                )
            except Exception:
                continue
            if rollouts is not None:
                return SearchResult(tree, rollouts, programs_tested, nodes_expanded)
            continue
        sequence, expanded = expand_hole(
            tree,
            path,
            dsl,
            costs,
            cost,
            sequence,
            frontier,
            int(config["max_depth"]),
        )
        nodes_expanded += expanded

    message = (
        f"no successful policy for {task['name']} within {budget} expansions "
        f"({programs_tested} complete programs tested)"
    )
    raise RuntimeError(message)


def pattern_from_tree(tree: dc.Delta) -> PatternNode:
    return PatternNode(
        symbol(tree),
        str(tree.type),
        tuple(pattern_from_tree(tail) for tail in tree.tails),
    )


def anti_unify(left: dc.Delta, right: dc.Delta) -> Pattern:
    variables: dict[tuple[Any, ...], PatternVariable] = {}

    def visit(a: dc.Delta, b: dc.Delta) -> Pattern:
        if tree_key(a) == tree_key(b):
            return pattern_from_tree(a)
        if (
            symbol(a) == symbol(b)
            and str(a.type) == str(b.type)
            and len(a.tails) == len(b.tails)
        ):
            return PatternNode(
                symbol(a),
                str(a.type),
                tuple(visit(x, y) for x, y in zip(a.tails, b.tails)),
            )
        key = (tree_key(a), tree_key(b), str(a.type))
        if key not in variables:
            variables[key] = PatternVariable(len(variables), str(a.type))
        return variables[key]

    return visit(left, right)


def pattern_variables(pattern: Pattern) -> tuple[PatternVariable, ...]:
    found: dict[int, PatternVariable] = {}

    def walk(node: Pattern) -> None:
        if isinstance(node, PatternVariable):
            found[node.index] = node
            return
        for child in node.children:
            walk(child)

    walk(pattern)
    return tuple(found[index] for index in sorted(found))


def fixed_node_count(pattern: Pattern) -> int:
    if isinstance(pattern, PatternVariable):
        return 0
    return 1 + sum(fixed_node_count(child) for child in pattern.children)


def pattern_node_count(pattern: Pattern) -> int:
    if isinstance(pattern, PatternVariable):
        return 1
    return 1 + sum(pattern_node_count(child) for child in pattern.children)


def pattern_key(pattern: Pattern) -> tuple[Any, ...]:
    if isinstance(pattern, PatternVariable):
        return ("var", pattern.index, pattern.output_type)
    return (
        "node",
        pattern.symbol,
        pattern.output_type,
        tuple(pattern_key(child) for child in pattern.children),
    )


def pattern_repr(pattern: Pattern) -> str:
    if isinstance(pattern, PatternVariable):
        return f"${pattern.index}:{pattern.output_type}"
    if not pattern.children:
        return pattern.symbol
    return f"({pattern.symbol} {' '.join(pattern_repr(c) for c in pattern.children)})"


def match_pattern(
    pattern: Pattern,
    tree: dc.Delta,
    bindings: dict[int, dc.Delta],
) -> bool:
    if str(tree.type) != pattern.output_type:
        return False
    if isinstance(pattern, PatternVariable):
        previous = bindings.get(pattern.index)
        if previous is None:
            bindings[pattern.index] = tree
            return True
        return tree_key(previous) == tree_key(tree)
    if pattern.symbol != symbol(tree) or len(pattern.children) != len(tree.tails):
        return False
    return all(
        match_pattern(child, tail, bindings)
        for child, tail in zip(pattern.children, tree.tails)
    )


def discover_skill(
    corpus: list[dc.Delta],
    config: dict[str, Any],
    known_patterns: set[tuple[Any, ...]],
) -> SkillCandidate | None:
    compression = config["compression"]
    candidates: dict[tuple[Any, ...], PatternNode] = {}
    for left_index, left in enumerate(corpus):
        for right in corpus[left_index + 1:]:
            pattern = anti_unify(left, right)
            if isinstance(pattern, PatternNode):
                candidates.setdefault(pattern_key(pattern), pattern)

    ranked: list[tuple[tuple[Any, ...], SkillCandidate]] = []
    for key, pattern in candidates.items():
        variables = pattern_variables(pattern)
        fixed_nodes = fixed_node_count(pattern)
        if key in known_patterns or not variables:
            continue
        if len(variables) > int(compression["max_arity"]):
            continue
        if len({variable.output_type for variable in variables}) != len(variables):
            continue
        if fixed_nodes < int(compression["min_fixed_nodes"]):
            continue

        matches = []
        for tree in corpus:
            bindings: dict[int, dc.Delta] = {}
            if match_pattern(pattern, tree, bindings):
                matches.append(bindings)
        support = len(matches)
        if support < int(compression["min_task_support"]):
            continue
        if any(
            len({tree_key(binding[variable.index]) for binding in matches}) < 2
            for variable in variables
        ):
            continue

        template_gain = support * (fixed_nodes - 1) - fixed_nodes
        strict_gain = (
            sum(dc.length(tree) for tree in corpus)
            - pattern_node_count(pattern)
            - support * (1 + len(variables))
        )
        if template_gain <= 0:
            continue
        candidate = SkillCandidate(
            pattern,
            support,
            fixed_nodes,
            len(variables),
            template_gain,
            strict_gain,
            matches,
        )
        rank = (
            -template_gain,
            -support,
            -fixed_nodes,
            len(variables),
            pattern_key(pattern),
        )
        ranked.append((rank, candidate))

    return min(ranked, default=(None, None), key=lambda item: item[0])[1]


def skill_name(pattern: PatternNode, index: int) -> str:
    if (
        pattern.symbol == "if"
        and len(pattern.children) == 3
        and isinstance(pattern.children[0], PatternNode)
        and isinstance(pattern.children[1], PatternNode)
    ):
        predicate = pattern.children[0].symbol
        action = pattern.children[1].symbol
        if predicate == f"goal_{action}":
            return f"go_{action}_then"
    return f"skill_{index}"


def pattern_to_delta(
    pattern: Pattern,
    primitives: dict[str, dc.Delta],
) -> dc.Delta:
    if isinstance(pattern, PatternVariable):
        return dc.Delta(f"${pattern.index}", type=pattern.output_type, isarg=True)
    primitive = primitives[pattern.symbol]
    return replace(
        primitive,
        tails=tuple(pattern_to_delta(child, primitives) for child in pattern.children),
    )


def build_invention(
    candidate: SkillCandidate,
    library: list[LibraryPrimitive],
    task_index: int,
) -> LibraryPrimitive:
    name = skill_name(candidate.pattern, task_index + 1)
    variables = pattern_variables(candidate.pattern)
    primitives = {symbol(entry.delta): entry.delta for entry in library}
    body = pattern_to_delta(candidate.pattern, primitives)
    return LibraryPrimitive(
        delta=dc.Delta(
            name,
            type=candidate.pattern.output_type,
            tailtypes=tuple(variable.output_type for variable in variables),
            hiddentail=body,
            repr_str=name,
        ),
        kind="invention",
        weight=1.0,
        definition=pattern_repr(candidate.pattern),
        learned_at=task_index,
    )


def rewrite_with_skill(
    tree: dc.Delta,
    pattern: PatternNode,
    invention: LibraryPrimitive,
) -> dc.Delta:
    bindings: dict[int, dc.Delta] = {}
    if match_pattern(pattern, tree, bindings):
        arguments = tuple(bindings[index] for index in sorted(bindings))
        return replace(invention.delta, tails=arguments)
    if not tree.tails:
        return tree
    return replace(
        tree,
        tails=tuple(rewrite_with_skill(t, pattern, invention) for t in tree.tails),
    )


def fit_weights(
    library: list[LibraryPrimitive],
    corpus: list[dc.Delta],
    alpha: float,
) -> None:
    counts = Counter(symbol(node) for tree in corpus for node in iter_program_nodes(tree))
    for primitive in library:
        primitive.weight = alpha + counts[symbol(primitive.delta)]


def rollout_payload(rollout: Rollout, case: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "case_index": index,
        "case": case,
        "success": rollout.success,
        "return": rollout.total_reward,
        "episode_steps": len(rollout.steps),
        "rgb_frames": rollout.rgb_frames,
        "terminal_frame_inferred": rollout.terminal_frame_inferred,
    }


def result_payload(result: SearchResult, task: dict[str, Any]) -> dict[str, Any]:
    trajectories = [
        rollout_payload(rollout, case, index)
        for index, (rollout, case) in enumerate(zip(result.rollouts, task["cases"]))
    ]
    return {
        "program": str(result.program),
        "program_nodes": dc.length(result.program),
        "programs_tested": result.programs_tested,
        "nodes_expanded": result.nodes_expanded,
        "success": all(trajectory["success"] for trajectory in trajectories),
        "episode_steps": sum(t["episode_steps"] for t in trajectories),
        "trajectories": trajectories,
    }


def make_task_environments(task: dict[str, Any]) -> list[MiniHackNavigation]:
    return [make_environment(case) for case in task["cases"]]


def close_environments(environments: list[MiniHackNavigation]) -> None:
    for environment in environments:
        environment.close()


def solve_task(
    task: dict[str, Any],
    library: list[LibraryPrimitive],
    config: dict[str, Any],
    prior_mode: str,
) -> SearchResult:
    environments = make_task_environments(task)
    try:
        return synthesize(environments, task, library, config, prior_mode)


def dynamic_obstacle_position(case: dict[str, Any]) -> tuple[int, int]:
    values = case.get("dynamic_gate", case.get("hazard"))
    if values is None:
        raise ValueError(f"dynamic task {case['name']} has no obstacle position")
    return int(values[0]), int(values[1])


def obstacle_active(case: dict[str, Any], time_step: int) -> bool:
    if "dynamic_gate" in case:
        return time_step >= int(case["gate_closes_at"])
    active_duration = int(case["active_duration"])
    if active_duration <= 0:
        return False
    period = int(case["period"])
    phase = int(case.get("phase", 0))
    return (time_step + phase) % period < active_duration


def obstacle_active_in_plan(
    case: dict[str, Any], arrival_time: int, planning_time: int
) -> bool:
    if "dynamic_gate" in case:
        return planning_time >= int(case["gate_closes_at"])
    return obstacle_active(case, arrival_time)


def walkable(case: dict[str, Any], position: tuple[int, int]) -> bool:
    width, height = map(int, case["size"])
    x, y = position
    if not (0 <= x < width and 0 <= y < height):
        return False
    if [x, y] in case.get("walls", []):
        return False
    return "corridor_row" not in case or y == int(case["corridor_row"])


def time_expanded_plan(
    case: dict[str, Any],
    start: tuple[int, int],
    start_time: int,
) -> tuple[list[int], int]:
    goal = tuple(map(int, case["goal"]))
    period = max(1, int(case["period"]))
    start_state = (start[0], start[1], start_time % period)
    frontier = [(0, 0, 0, start_state)]
    best_cost = {start_state: 0}
    parent: dict[tuple[int, int, int], tuple[tuple[int, int, int], int]] = {}
    sequence = 0
    expanded = 0

    while frontier:
        _, cost, _, state = heapq.heappop(frontier)
        x, y, phase = state
        if (x, y) == goal:
            actions = []
            while state != start_state:
                previous, action = parent[state]
                actions.append(action)
                state = previous
            actions.reverse()
            return actions, expanded
        if cost != best_cost.get(state):
            continue
        expanded += 1
        absolute_next_time = start_time + cost + 1
        next_phase = (phase + 1) % period
        for action, (dx, dy) in enumerate(ACTION_DELTAS):
            candidate = (x + dx, y + dy)
            if not walkable(case, candidate):
                continue
            if (
                candidate == dynamic_obstacle_position(case)
                and obstacle_active_in_plan(case, absolute_next_time, start_time)
            ):
                continue
            next_state = (candidate[0], candidate[1], next_phase)
            next_cost = cost + 1
            if next_cost >= best_cost.get(next_state, 1_000_000):
                continue
            best_cost[next_state] = next_cost
            parent[next_state] = (state, action)
            sequence += 1
            heuristic = abs(candidate[0] - goal[0]) + abs(candidate[1] - goal[1])
            heapq.heappush(
                frontier,
                (next_cost + heuristic, next_cost, sequence, next_state),
            )
    raise RuntimeError(f"no time-expanded path for dynamic task {case['name']}")


def run_dynamic_strategy(
    case: dict[str, Any], strategy: str, seed: int
) -> dict[str, Any]:
    environment = make_environment(case)
    try:
        observation, _ = environment.reset(seed=seed)
        frame = ascii_frame(observation)
        rendered = rgb_frame(observation)
        if frame is None or rendered is None:
            raise ValueError("dynamic room did not produce renderable observations")
        frames = [frame]
        rgb_frames = [rendered]
        hazard_states = [obstacle_active(case, 0)]
        trace = []
        start_values = list(map(int, case["start"]))
        goal_values = list(map(int, case["goal"]))
        position = (start_values[0], start_values[1])
        goal = (goal_values[0], goal_values[1])
        planning_expansions = 0
        replans = 0
        plan: list[int] = []
        success = False
        collision = False
        plan_invalidated = False
        failure_reason = None
        initial_plan: list[str] = []

        if strategy == "one_shot_astar":
            plan, expanded = time_expanded_plan(case, position, 0)
            planning_expansions += expanded
            replans = 1
            initial_plan = [ACTION_NAMES[action] for action in plan]

        for step_index in range(int(case["max_steps"])):
            if strategy == "static_program":
                action = 1 if position[0] < goal[0] else 4
            elif strategy == "one_shot_astar":
                action = plan.pop(0) if plan else 4
            elif strategy == "replanning_astar":
                plan, expanded = time_expanded_plan(case, position, step_index)
                planning_expansions += expanded
                replans += 1
                action = plan[0] if plan else 4
            else:
                raise ValueError(f"unknown dynamic strategy: {strategy}")

            dx, dy = ACTION_DELTAS[action]
            proposed = (position[0] + dx, position[1] + dy)
            next_time = step_index + 1
            active = obstacle_active(case, next_time)
            if proposed == dynamic_obstacle_position(case) and active:
                collision = "hazard" in case
                plan_invalidated = "dynamic_gate" in case
                failure_reason = (
                    "plan invalidated by unexpected gate closure"
                    if plan_invalidated
                    else "entered active blinking hazard"
                )
                frames.append(frames[-1])
                rgb_frames.append(rgb_frames[-1])
                hazard_states.append(active)
                trace.append({
                    "step": step_index,
                    "player": list(position),
                    "goal": list(goal),
                    "action": ACTION_NAMES[action],
                    "reward": -1.0,
                    "collision": collision,
                    "plan_invalidated": plan_invalidated,
                    "failure_reason": failure_reason,
                    "hazard_active": active,
                    "replanned": strategy == "replanning_astar",
                })
                break

            observation, reward, terminated, truncated, _ = environment.step(action)
            if walkable(case, proposed):
                position = proposed
            next_frame = ascii_frame(observation)
            next_render = rgb_frame(observation)
            frames.append(next_frame if next_frame is not None else frames[-1])
            rgb_frames.append(next_render if next_render is not None else rgb_frames[-1])
            hazard_states.append(active)
            trace.append({
                "step": step_index,
                "player": list(position),
                "goal": list(goal),
                "action": ACTION_NAMES[action],
                "reward": float(reward),
                "collision": False,
                "plan_invalidated": False,
                "failure_reason": None,
                "hazard_active": active,
                "replanned": strategy == "replanning_astar",
            })
            if terminated or truncated:
                success = bool(terminated and reward > 0)
                break

        if not success and failure_reason is None:
            failure_reason = "episode limit reached without a valid plan"

        return {
            "name": strategy,
            "program": {
                "static_program": "move east toward known goal",
                "one_shot_astar": "A* over (x, y, time mod period), planned once",
                "replanning_astar": "A* over (x, y, time mod period), replanned each step",
            }[strategy],
            "success": success,
            "collision": collision,
            "plan_invalidated": plan_invalidated,
            "failure_reason": failure_reason,
            "initial_plan": initial_plan,
            "episode_steps": len(trace),
            "programs_tested": 0,
            "nodes_expanded": planning_expansions,
            "program_nodes": None,
            "planning_expansions": planning_expansions,
            "replans": replans,
            "trajectories": [{
                "case_index": 0,
                "case": case,
                "success": success,
                "return": 1.0 if success else -1.0 if collision else 0.0,
                "episode_steps": len(trace),
                "trajectory": trace,
                "frames": frames,
                "rgb_frames": rgb_frames,
                "hazard_states": hazard_states,
                "terminal_frame_inferred": False,
            }],
        }
    finally:
        environment.close()


def run_dynamic_evaluation(config: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for index, case in enumerate(config.get("dynamic_tasks", [])):
        strategies = [
            run_dynamic_strategy(case, strategy, int(config["seed"]))
            for strategy in (
                "static_program",
                "one_shot_astar",
                "replanning_astar",
            )
        ]
        output.append({
            "id": f"dynamic-{index + 1}",
            "index": index,
            "split": "dynamic",
            "name": str(case["name"]),
            "case": case,
            "strategies": strategies,
            "failure_mode": {
                "static_failed": not strategies[0]["success"],
                "one_shot_succeeded": strategies[1]["success"],
                "replanning_succeeded": strategies[2]["success"],
                "replanning_fixed_stale_plan": (
                    not strategies[1]["success"] and strategies[2]["success"]
                ),
            },
        })
    return output


def candidate_payload(candidate: SkillCandidate, invention: LibraryPrimitive) -> dict[str, Any]:
    return {
        "name": symbol(invention.delta),
        "signature": list(invention.delta.tailtypes),
        "definition": invention.definition,
        "support": candidate.support,
        "fixed_nodes": candidate.fixed_nodes,
        "arity": candidate.arity,
        "template_gain": candidate.template_gain,
        "strict_ast_gain": candidate.strict_ast_gain,
    }


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    base_library = initial_library(config)
    library = clone_library(base_library)
    initial = library_snapshot(library, str(config["prior_mode"]))
    corpus: list[dc.Delta] = []
    rewritten_corpus: list[dc.Delta] = []
    known_patterns: set[tuple[Any, ...]] = set()
    training = []

    for task_index, task in enumerate(config["train_tasks"]):
        before = library_snapshot(library, str(config["prior_mode"]))
        result = solve_task(task, library, config, str(config["prior_mode"]))
        corpus.append(result.program)
        rewritten_corpus.append(result.program)

        compression_phase: dict[str, Any] = {
            "status": "skipped",
            "reason": "need at least two distinct solved programs",
        }
        prior_phase: dict[str, Any] = {
            "status": "skipped",
            "reason": "library unchanged",
        }
        invention_payload = None

        if bool(config["compression"].get("enabled", True)) and len(corpus) >= 2:
            candidate = discover_skill(corpus, config, known_patterns)
            if candidate is None:
                compression_phase = {
                    "status": "skipped",
                    "reason": "no non-trivial typed anti-unification passed filters",
                }
            else:
                invention = build_invention(candidate, library, task_index)
                known_patterns.add(pattern_key(candidate.pattern))
                library.append(invention)
                rewritten_corpus = [
                    rewrite_with_skill(tree, candidate.pattern, invention)
                    for tree in rewritten_corpus
                ]
                alpha = float(config["compression"]["prior_alpha"])
                fit_weights(library, rewritten_corpus, alpha)
                invention_payload = candidate_payload(candidate, invention)
                compression_phase = {
                    "status": "complete",
                    "input_programs": [str(tree) for tree in corpus],
                    "output": invention_payload,
                    "rewritten_corpus": [str(tree) for tree in rewritten_corpus],
                }
                prior_phase = {
                    "status": "complete",
                    "alpha": alpha,
                    "counts_from": [str(tree) for tree in rewritten_corpus],
                }

        training.append({
            "id": f"train-{task_index + 1}",
            "index": task_index,
            "split": "train",
            "name": str(task["name"]),
            "explore": {
                "status": "complete",
                **result_payload(result, task),
            },
            "compress": compression_phase,
            "update_prior": prior_phase,
            "library_before": before,
            "library_after": library_snapshot(library, str(config["prior_mode"])),
            "invention": invention_payload,
        })

    alpha = float(config["compression"]["prior_alpha"])
    base_fitted = clone_library(base_library)
    fit_weights(base_fitted, corpus, alpha)
    learned_fitted = clone_library(library)
    learned_uniform = clone_library(library)
    heldout = []

    conditions = (
        ("frozen_uniform", clone_library(base_library), "uniform"),
        ("frozen_fitted", base_fitted, "weighted"),
        ("learned_uniform", learned_uniform, "uniform"),
        ("learned_fitted", learned_fitted, "weighted"),
    )
    for task_index, task in enumerate(config["heldout_tasks"]):
        condition_results = []
        for name, condition_library, prior_mode in conditions:
            result = solve_task(task, condition_library, config, prior_mode)
            condition_results.append({
                "name": name,
                "prior_mode": prior_mode,
                "library": library_snapshot(condition_library, prior_mode),
                **result_payload(result, task),
            })
        frozen = next(c for c in condition_results if c["name"] == "frozen_uniform")
        learned = next(c for c in condition_results if c["name"] == "learned_fitted")
        heldout.append({
            "id": f"test-{task_index + 1}",
            "index": task_index,
            "split": "test",
            "name": str(task["name"]),
            "conditions": condition_results,
            "generalization": {
                "all_cases_success": learned["success"],
                "uses_parameterized_skill": "go_east_then" in learned["program"],
                "learned_program": learned["program"],
                "frozen_program": frozen["program"],
                "learned_expansions": learned["nodes_expanded"],
                "frozen_expansions": frozen["nodes_expanded"],
                "expansion_ratio": (
                    learned["nodes_expanded"] / frozen["nodes_expanded"]
                    if frozen["nodes_expanded"]
                    else None
                ),
            },
    dynamic = run_dynamic_evaluation(config)
    return {
        "config": config,
        "initial_library": initial,
        "training": training,
        "heldout": heldout,
        "dynamic": dynamic,
        "summary": {
            "training_tasks": len(training),
            "heldout_tasks": len(heldout),
            "skills_discovered": sum(step["invention"] is not None for step in training),
            "final_library_size": len(library),
            "generalization_passed": all(
                task["generalization"]["all_cases_success"]
            ),
            "dynamic_failure_displayed": any(
                task["failure_mode"]["replanning_fixed_stale_plan"]
                for task in dynamic
            ),
        },
    }


def format_frames(payload: dict[str, Any]) -> str:
    sections = []
    for step in payload["training"]:
        result = step["explore"]
        for trajectory in result["trajectories"]:
            sections.append(format_trajectory(step["name"], result["program"], trajectory))
    for task in payload["heldout"]:
        learned = next(c for c in task["conditions"] if c["name"] == "learned_fitted")
    for task in payload["dynamic"]:
        for strategy in task["strategies"]:
            sections.append(
                format_trajectory(task["name"], strategy["program"], strategy["trajectories"][0])
            )
    return ("\n\n" + "=" * 72 + "\n\n").join(sections) + "\n"


def format_trajectory(name: str, program: str, trajectory: dict[str, Any]) -> str:
    sections = [
        f"task: {name}; case {trajectory['case_index'] + 1}",
        f"program: {program}",
        f"frame 0: start\n{trajectory['frames'][0]}",
    ]
    for index, (transition, frame) in enumerate(
        zip(trajectory["trajectory"], trajectory["frames"][1:]), start=1
    ):
        inference_note = (
            " [inferred terminal frame]"
            if index == len(trajectory["frames"]) - 1
            and trajectory["terminal_frame_inferred"]
            else ""
        )
        sections.append(
            f"frame {index}: after {transition['action']} "
            f"(reward={transition['reward']}){inference_note}\n{frame}"
        )
    return "\n\n".join(sections)


def render_html(payload: dict[str, Any]) -> str:
    template = HTML_TEMPLATE_PATH.read_text()
    embedded_data = json.dumps(payload).replace("</", "<\\/")
    if "__MINIHACK_DATA__" not in template:
        raise ValueError(f"missing data placeholder in {HTML_TEMPLATE_PATH}")
    return template.replace("__MINIHACK_DATA__", embedded_data)


def save_outputs(payload: dict[str, Any]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ = RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    _ = FRAME_PATH.write_text(format_frames(payload))
    _ = HTML_PATH.write_text(render_html(payload))


def print_summary(payload: dict[str, Any], config_path: Path) -> None:
    print(f"config:      {config_path}")
    for step in payload["training"]:
        explore = step["explore"]
        skill = step["invention"]["name"] if step["invention"] else "none"
        print(
            f"train {step['index'] + 1}:     {step['name']}: "
            f"program={explore['program']}, expanded={explore['nodes_expanded']}, "
            f"skill={skill}"
        )
    for task in payload["heldout"]:
        generalization = task["generalization"]
        print(
            f"held-out:    {task['name']}: learned={generalization['learned_program']}, "
            f"expanded learned/frozen={generalization['learned_expansions']}/"
            f"{generalization['frozen_expansions']}"
    for task in payload["dynamic"]:
        results = {strategy["name"]: strategy for strategy in task["strategies"]}
        print(
            f"dynamic:     {task['name']}: static="
            f"{results['static_program']['success']}, one-shot="
            f"{results['one_shot_astar']['success']}, replanning="
            f"{results['replanning_astar']['success']}"
        )


def run_once(config_path: Path, render: bool = False) -> dict[str, Any]:
    payload = run_experiment(load_config(config_path))
    save_outputs(payload)
    print_summary(payload, config_path)
    if render:
        print(format_frames(payload), end="")
    return payload


class DemoServer(ThreadingHTTPServer):
    payload: dict[str, Any]
    config_path: Path
    run_lock: threading.Lock

    def __init__(
        self,
        address: tuple[str, int],
        payload: dict[str, Any],
        config_path: Path,
    ):
        self.payload = payload
        self.config_path = config_path
        self.run_lock = threading.Lock()
        super().__init__(address, DemoHandler)


class DemoHandler(BaseHTTPRequestHandler):
    @property
    def demo_server(self) -> DemoServer:
        return cast(DemoServer, self.server)

    def send_content(self, content: bytes, content_type: str, status: int = 200) -> None:
        _ = self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, value: Any, status: int = 200) -> None:
        self.send_content(
            json.dumps(value).encode(),
            "application/json; charset=utf-8",
            status,
        )

    def do_GET(self) -> None:
        if self.path == "/":
            self.send_content(
                render_html(self.demo_server.payload).encode(),
                "text/html; charset=utf-8",
            )
            return
        if self.path == "/api/state":
            self.send_json(self.demo_server.payload)
            return
        self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path != "/api/run":
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        if not self.demo_server.run_lock.acquire(blocking=False):
            self.send_json(
                {"error": "an experiment is already running"},
                HTTPStatus.CONFLICT,
            )
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            config = cast(dict[str, Any], json.loads(self.rfile.read(length)))
            payload = run_experiment(config)
            _ = self.demo_server.config_path.write_text(
                json.dumps(config, indent=2) + "\n"
            )
            save_outputs(payload)
            self.demo_server.payload = payload
            print_summary(payload, self.demo_server.config_path)
            self.send_json(payload)
        except Exception as error:
            self.send_json({"error": str(error)}, HTTPStatus.UNPROCESSABLE_ENTITY)
        finally:
            self.demo_server.run_lock.release()

    def log_message(self, format: str, *args: Any) -> None:
        print(f"http: {format % args}")


def serve(options: Cli, payload: dict[str, Any]) -> None:
    server = DemoServer((options.host, options.port), payload, options.config)
    url = f"http://{options.host}:{server.server_port}/"
    print(f"demo:        {url}")
    if options.open_browser:
        _ = webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping server")
    finally:
        server.server_close()


def main() -> None:
    options = parse(Cli)
    payload = run_once(options.config, render=options.render)
    if options.mode is Mode.serve:
        serve(options, payload)


if __name__ == "__main__":
    main()
