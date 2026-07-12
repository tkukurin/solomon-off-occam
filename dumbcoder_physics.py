"""Library-learning experiments over physics formulas, dreamcoder-style.

1. Curriculum: solve kinetic, rotational, spring, potential, work, and
   mechanical energy in sequence.

   frozen base DSL (uniform search) compared vs DreamCoder's full sleep phase:
   repeated solution subtrees are packaged as inventions and production
   probabilities are re-estimated from the rewritten corpus
   (Ellis et al., PLDI 2021; Dechter et al., 2013).

2. Typing: distinct physical type tags act as dim-analysis pruning. Collapsing
   every quantity into one type blows the search up by orders of magnitude.

   (cf. AI Feynman; Udrescu & Tegmark, 2020)

3. Priors: types gone, unigram prior fitted on previously solved programs guides
   best-first enumeration. misleading prior worse than uniform.

   (cf. DeepCoder; Balog et al., 2017)

4. Library bloat: distractor inventions raise the branching factor.
   NFL, compression costs.

"""

import heapq
from collections import Counter
from dataclasses import dataclass, replace
from itertools import accumulate
from math import log
from typing import cast

import matplotlib

matplotlib.use("Agg")  # drop this line if you want an interactive window
import matplotlib.pyplot as plt

import dumbcoder as dc

MASS = "mass"
VELOCITY = "velocity"
GRAVITY = "gravity"
HEIGHT = "height"
VELOCITY_SQUARED = "velocity_squared"
MASS_VELOCITY_SQUARED = "mass_velocity_squared"
MASS_GRAVITY = "mass_gravity"
ENERGY = "energy"
TOTAL_ENERGY = "total_energy"
DIMENSIONLESS = "dimensionless"
QUANTITY = "quantity"  # the one type of the untyped world

UNTYPED_BUDGET = 50_000
PRIOR_BUDGET = 60_000
PLOT_PATH = "dumbcoder_experiments.png"


@dataclass
class Task:
    name: str
    arguments: tuple
    output_type: str
    examples: tuple


def argument(name: str, tag: str) -> dc.Delta:
    return dc.Delta(f"${name}", type=tag, isarg=True)


def add_primitive() -> dc.Delta:
    return dc.Delta(head=lambda left, right: left + right, type=TOTAL_ENERGY,
                    tailtypes=(ENERGY, ENERGY), repr_str="add")


def base_primitives() -> list[dc.Delta]:
    # deliberately fine-grained: squaring is an explicit self-product and
    # halving an explicit division by the constant 2, so raw solutions are
    # deep (kinetic: 7 nodes, mechanical: 13) and compression has room to pay
    return [
        dc.Delta(head=lambda v, w: v * w, type=VELOCITY_SQUARED,
                 tailtypes=(VELOCITY, VELOCITY), repr_str="multiply"),
        dc.Delta(head=lambda m, v2: m * v2, type=MASS_VELOCITY_SQUARED,
                 tailtypes=(MASS, VELOCITY_SQUARED), repr_str="multiply"),
        dc.Delta(head=1.0, type=DIMENSIONLESS, repr_str="1"),
        dc.Delta(head=2.0, type=DIMENSIONLESS, repr_str="2"),
        dc.Delta(head=3.0, type=DIMENSIONLESS, repr_str="3"),
        dc.Delta(head=lambda value, by: value / by, type=ENERGY,
                 tailtypes=(MASS_VELOCITY_SQUARED, DIMENSIONLESS), repr_str="divide"),
        dc.Delta(head=lambda m, g: m * g, type=MASS_GRAVITY,
                 tailtypes=(MASS, GRAVITY), repr_str="multiply"),
        dc.Delta(head=lambda mg, h: mg * h, type=ENERGY,
                 tailtypes=(MASS_GRAVITY, HEIGHT), repr_str="multiply"),
        # sign-convention op: a realistic library carries productions most
        # tasks never need, so each extra search level costs real branching
        dc.Delta(head=lambda value: -value, type=ENERGY,
                 tailtypes=(ENERGY,), repr_str="negate"),
        add_primitive(),
    ]


def untyped_primitives() -> list[dc.Delta]:
    # the same operations with every type tag collapsed into QUANTITY
    return [
        dc.Delta(head=lambda left, right: left * right, type=QUANTITY,
                 tailtypes=(QUANTITY, QUANTITY), repr_str="multiply"),
        dc.Delta(head=1.0, type=QUANTITY, repr_str="1"),
        dc.Delta(head=2.0, type=QUANTITY, repr_str="2"),
        dc.Delta(head=3.0, type=QUANTITY, repr_str="3"),
        dc.Delta(head=lambda value, by: value / by, type=QUANTITY,
                 tailtypes=(QUANTITY, QUANTITY), repr_str="divide"),
        dc.Delta(head=lambda left, right: left + right, type=QUANTITY,
                 tailtypes=(QUANTITY, QUANTITY), repr_str="add"),
        dc.Delta(head=lambda value: -value, type=QUANTITY,
                 tailtypes=(QUANTITY,), repr_str="negate"),
    ]


def curriculum_tasks() -> tuple:
    # Rotational inertia, angular velocity, stiffness, and displacement reuse
    # the mass/velocity type slots: I*w^2 and k*x^2 fill the same dimensional
    # roles in the 1/2*a*b^2 template as m*v^2, which is what lets a single
    # invention cover all three tasks.
    return (
        Task("kinetic 1/2mv2",
             (argument("m", MASS), argument("v", VELOCITY)), ENERGY,
             (((2.0, 3.0), 9.0), ((4.0, 5.0), 50.0), ((8.0, 2.0), 16.0))),
        Task("rotational 1/2Iw2",
             (argument("I", MASS), argument("w", VELOCITY)), ENERGY,
             (((1.5, 2.0), 3.0), ((3.0, 4.0), 24.0), ((0.5, 6.0), 9.0))),
        Task("spring 1/2kx2",
             (argument("k", MASS), argument("x", VELOCITY)), ENERGY,
             (((100.0, 0.2), 2.0), ((50.0, 0.5), 6.25), ((200.0, 0.1), 1.0))),
        Task("potential mgh",
             (argument("m", MASS), argument("g", GRAVITY), argument("h", HEIGHT)),
             ENERGY,
             (((2.0, 9.81, 4.0), 78.48), ((4.0, 9.81, 3.0), 117.72),
              ((8.0, 1.62, 10.0), 129.6))),
        Task("work mad",  # W = F*d = (m*a)*d, a structural twin of mgh
             (argument("m", MASS), argument("a", GRAVITY), argument("d", HEIGHT)),
             ENERGY,
             (((3.0, 2.0, 5.0), 30.0), ((1.5, 4.0, 2.0), 12.0),
              ((2.0, 9.81, 1.0), 19.62))),
        Task("mechanical KE+PE",
             (argument("m", MASS), argument("v", VELOCITY),
              argument("g", GRAVITY), argument("h", HEIGHT)), TOTAL_ENERGY,
             (((2.0, 3.0, 9.81, 4.0), 87.48), ((4.0, 5.0, 9.81, 3.0), 167.72),
              ((8.0, 2.0, 1.62, 10.0), 145.6))),
    )


def untyped_variant(task: Task) -> Task:
    arguments = tuple(
        dc.Delta(arg.head, type=QUANTITY, isarg=True) for arg in task.arguments)
    return Task(task.name, arguments, QUANTITY, task.examples)


def substitute(tree: dc.Delta, bindings: dict) -> dc.Delta:
    """Bind arguments by head name -- needed in the untyped world, where every
    argument shares one type and replace_hidden could not tell $m from $v."""
    if tree.isarg:
        return dc.Delta(bindings[tree.head], type=tree.type)
    if not tree.tails:
        return tree
    return replace(tree, tails=tuple(substitute(tail, bindings) for tail in tree.tails))


def _evaluate_example(tree, values, expected, arguments, by_head) -> bool:
    if by_head:
        mapping = {arg.head: val for arg, val in zip(arguments, values)}
        applied = substitute(tree, mapping)
    else:
        applied = tree
        for arg, val in zip(arguments, values):
            applied = dc.replace_hidden(applied, arg, dc.Delta(val, type=arg.type))
    try:
        return abs(cast(float, applied()) - expected) <= 1e-9
    except (ZeroDivisionError, OverflowError):
        return False


def matches_examples(tree, examples, arguments, by_head=False) -> bool:
    return all(
        _evaluate_example(tree, vals, exp, arguments, by_head)
        for vals, exp in examples
    )


def node_key(node: dc.Delta) -> str:
    return "arg" if node.isarg else str(node.repr_str or node.head)


def iter_nodes(tree):
    stack = [tree]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.tails or ())


def _expand_solve_node(
    tree, path, dsl, costs, cost, pushed, expanded_count, frontier
):
    hole = tree
    for index in path:
        hole = hole.tails[index]

    for primitive_index in dsl.bytype.get(hole.type, ()):
        primitive = dsl[primitive_index]
        tails = tuple(
            dc.Delta(f"<>:{t}", type=t, ishole=True)
            for t in primitive.tailtypes or ()
        )
        expanded = replace(primitive, tails=tails)
        step = 1.0 if costs is None else costs[node_key(primitive)]

        heapq.heappush(
            frontier,
            (cost + step, pushed, dc._replace_at_path(tree, path, expanded))
        )
        pushed += 1
        expanded_count += 1

    return pushed, expanded_count


def solve(examples, dsl, arguments, output_type,
          costs=None, budget=None, by_head=False):
    """Best-first enumeration over typed trees. Uniform costs (costs=None)
    reduce to the breadth-first search of the base example; otherwise each
    expansion pays the -log prior probability of the chosen primitive."""
    root = dc.Delta(f"<>:{output_type}", type=output_type, ishole=True)
    frontier = [(0.0, 0, root)]
    pushed = 1
    tested = 0
    expanded_count = 0

    while frontier:
        if budget is not None and expanded_count >= budget:
            return None, tested, expanded_count
        cost, _, tree = heapq.heappop(frontier)
        path = dc._find_hole(tree)
        
        if path is None:
            tested += 1
            if matches_examples(tree, examples, arguments, by_head=by_head):
                return tree, tested, expanded_count
            continue

        pushed, expanded_count = _expand_solve_node(
            tree, path, dsl, costs, cost, pushed, expanded_count, frontier
        )

    return None, tested, expanded_count




def canonical(tree: dc.Delta) -> str:
    """Structural key with arguments abstracted to their type, so 1/2*m*v^2
    and 1/2*I*w^2 count as the same reusable pattern."""
    if tree.isarg:
        return f"?{tree.type}"
    label = str(tree.repr_str or tree.head)
    if not tree.tails:
        return label
    return f"{label}({', '.join(canonical(tail) for tail in tree.tails)})"


def subtrees(tree):
    if tree.tails:
        yield tree
        for tail in tree.tails:
            yield from subtrees(tail)


def invention_arguments(body: dc.Delta) -> tuple:
    """One argument per type, in order of first appearance."""
    seen: dict = {}

    def walk(node):
        if node.isarg and node.type not in seen:
            seen[node.type] = node
        for tail in node.tails or ():
            walk(tail)

    walk(body)
    return tuple(seen.values())


def _compute_gain(count: int, body: dc.Delta, known: set, key: str) -> int:
    if count < 2 or key in known:
        return 0
    if all(tail.isarg for tail in body.tails):
        return 0
    if not invention_arguments(body):
        return 0
    return (count - 1) * (dc.length(body) - 1)


def discover(corpus, known: set, name: str):
    """Promote the repeated subtree with the best compression gain -- a
    greedy, unigram version of DreamCoder's sleep/compression phase."""
    counts: Counter = Counter()
    samples: dict = {}
    for tree in corpus:
        for subtree in subtrees(tree):
            key = canonical(subtree)
            counts[key] += 1
            samples.setdefault(key, subtree)

    best_key, best_gain = None, 0
    for key, count in counts.items():
        gain = _compute_gain(count, samples[key], known, key)
        if gain > best_gain:
            best_key, best_gain = key, gain

    if best_key is None:
        return None
    known.add(best_key)
    body = samples[best_key]
    return dc.Delta(head=name, type=body.type,
                    tailtypes=tuple(arg.type for arg in invention_arguments(body)),
                    hiddentail=body, repr_str=name)


def rewrite(tree: dc.Delta, invention: dc.Delta) -> dc.Delta:
    """Fold occurrences of the invention body back into the corpus so later
    compression rounds don't rediscover fragments of it."""
    if tree.tails and canonical(tree) == canonical(invention.hiddentail):
        return replace(invention, tails=invention_arguments(tree))
    if not tree.tails:
        return tree
    return replace(tree, tails=tuple(rewrite(tail, invention) for tail in tree.tails))




def costs_from_counts(counts, primitives) -> dict:
    """Laplace-smoothed unigram production costs (-log p); the probability
    mass assigned to arguments is split evenly between them."""
    keys = {node_key(primitive) for primitive in primitives}
    normalizer = sum(counts.get(key, 0) + 0.5 for key in keys)
    n_arguments = max(1, sum(primitive.isarg for primitive in primitives))
    costs = {}
    for key in keys:
        probability = (counts.get(key, 0) + 0.5) / normalizer
        if key == "arg":
            probability /= n_arguments
        costs[key] = -log(probability)
    return costs


def misleading_counts(counts, primitives) -> Counter:
    keys = {node_key(primitive) for primitive in primitives}
    peak = max(counts.get(key, 0) for key in keys)
    return Counter({key: peak - counts.get(key, 0) for key in keys})




def _run_curriculum_step(
    task, index, library, corpus, known, usage,
    frozen, inventions, events, log
):
    dsl = dc.Deltas([*library, *task.arguments])
    costs = None if usage is None else costs_from_counts(usage, dsl.core)
    tree, _, expanded = solve(task.examples, dsl, task.arguments,
                              task.output_type, costs=costs)
    if tree is None: raise ValueError("Tree cannot be None")

    log(f"{task.name}: {tree}")
    log(f"  frozen={frozen[index]}, growing={expanded} expansions, "
        f"{dc.length(tree)} nodes")
    corpus.append(tree)
    
    invention = discover(corpus, known, f"f{len(inventions) + 1}")
    if invention is not None:
        signature = ", ".join(invention.tailtypes)
        log(f"  invented {invention.repr_str}({signature})"
            f" = {invention.hiddentail}")
        inventions.append(invention)
        library.append(invention)
        corpus = [rewrite(t, invention) for t in corpus]
        events.append((index, invention.repr_str))
        
    usage = Counter(node_key(node)
                    for t in corpus for node in iter_nodes(t))
    return expanded, usage, corpus


def run_curriculum(tasks, log=lambda _: None):
    log("== curriculum: frozen DSL vs growing library + refitted prior ==")
    frozen = []
    for task in tasks:
        dsl = dc.Deltas([*base_primitives(), *task.arguments])
        tree, _, expanded = solve(task.examples, dsl, task.arguments, task.output_type)
        if tree is None: raise ValueError("Tree cannot be None")
        frozen.append(expanded)

    # The growing condition is the full sleep phase: compress the corpus into
    # inventions AND re-estimate production probabilities from the rewritten
    # corpus. Without the refitted prior the library would only add branching
    # at these shallow depths -- the grammar weights are what make reuse cheap.
    growing, events, inventions = [], [], []
    library = base_primitives()
    corpus = []
    known = set()
    usage = None  # None on the first task: nothing solved yet, prior is uniform
    for index, task in enumerate(tasks):
        expanded, usage, corpus = _run_curriculum_step(
            task, index, library, corpus, known, usage,
            frozen, inventions, events, log
        )
        growing.append(expanded)
        
    return frozen, growing, inventions, events


def run_typing_ablation(tasks, frozen, log=lambda _: None):
    log("\n== typing ablation: physical types as a hard prior ==")
    picks = (0, 3, 5)
    labels = tuple(tasks[index].name.split()[0] for index in picks)
    typed = tuple(frozen[index] for index in picks)
    untyped = []
    for label, index in zip(labels, picks):
        task = untyped_variant(tasks[index])
        dsl = dc.Deltas([*untyped_primitives(), *task.arguments])
        tree, _, expanded = solve(task.examples, dsl, task.arguments, QUANTITY,
                                  budget=UNTYPED_BUDGET, by_head=True)
        untyped.append((expanded, tree is not None))
        note = f"program={tree}" if tree is not None else "budget hit"
        log(f"  {label}: typed={typed[len(untyped) - 1]}, untyped={expanded} ({note})")
    return labels, typed, untyped


def run_prior_comparison(tasks, log=lambda _: None):
    log("\n== priors: guiding untyped search for potential energy ==")
    ops = untyped_primitives()

    corpus = []  # warm-up: the structural twin (work), solved uniformly
    for warmup in (untyped_variant(tasks[4]),):
        dsl = dc.Deltas([*ops, *warmup.arguments])
        tree, _, _ = solve(warmup.examples, dsl, warmup.arguments, QUANTITY,
                           budget=PRIOR_BUDGET, by_head=True)
        if tree is None: raise ValueError("Tree cannot be None")
        corpus.append(tree)

    target = untyped_variant(tasks[3])
    dsl = dc.Deltas([*ops, *target.arguments])
    counts = Counter(node_key(node) for tree in corpus for node in iter_nodes(tree))
    conditions = (
        ("misleading", costs_from_counts(misleading_counts(counts, dsl.core), dsl.core)),
        ("uniform", None),
        ("fitted", costs_from_counts(counts, dsl.core)),
    )
    results = []
    for label, costs in conditions:
        tree, _, expanded = solve(target.examples, dsl, target.arguments, QUANTITY,
                                  costs=costs, budget=PRIOR_BUDGET, by_head=True)
        note = f"program={tree}" if tree is not None else "budget hit"
        log(f"  {label:>10}: expanded={expanded} ({note})")
        results.append((label, expanded, tree is not None))
    return results


def make_distractors(count: int) -> list[dc.Delta]:
    tailtype_options = ((MASS, HEIGHT), (GRAVITY, HEIGHT), (MASS, GRAVITY))
    return [dc.Delta(head=lambda left, right, k=float(index): left * right + k,
                     type=ENERGY, tailtypes=tailtype_options[index % 3],
                     repr_str=f"junk{index}")
            for index in range(count)]


def run_library_bloat(inventions, mechanical: Task, log=lambda _: None):
    log("\n== library bloat: distractor inventions on mechanical energy ==")
    if [invention.tailtypes for invention in inventions] != [
        (MASS, VELOCITY), (MASS, GRAVITY, HEIGHT),
    ]:
        raise ValueError(
            "curriculum discovery should have produced the KE and PE inventions"
        )
    counts = (0, 2, 4, 8, 16)
    expansions = []
    for count in counts:
        dsl = dc.Deltas([add_primitive(), *inventions, *make_distractors(count),
                         *mechanical.arguments])
        tree, _, expanded = solve(mechanical.examples, dsl, mechanical.arguments,
                                  TOTAL_ENERGY)
        if tree is None: raise ValueError("Tree cannot be None")
        expansions.append(expanded)
        log(f"  {count:>2} distractors: expanded={expanded}")
    return counts, expansions


# removed plotting separator

def _plot_curriculum(axis, tasks, frozen, growing, events):
    steps = range(1, len(tasks) + 1)
    axis.plot(steps, list(accumulate(frozen)), marker="o",
              label="frozen DSL, uniform prior")
    cumulative = list(accumulate(growing))
    axis.plot(steps, cumulative, marker="o",
              label="growing library, refitted prior")
    for index, name in events:
        axis.annotate(f"+{name}", (index + 1, cumulative[index]),
                      textcoords="offset points", xytext=(8, -12))
    axis.set_xticks(list(steps))
    axis.set_xticklabels([task.name.split()[0] for task in tasks], rotation=15)
    axis.set_ylabel("cumulative expansions")
    axis.set_title("curriculum: inventions amortize search")
    axis.legend()


def _plot_ablation(axis, ablation):
    labels, typed, untyped = ablation
    positions = range(len(labels))
    axis.bar([p - 0.2 for p in positions], typed, width=0.4, label="typed")
    bars = axis.bar([p + 0.2 for p in positions],
                    [value for value, _ in untyped], width=0.4, label="untyped")
    for bar, (_, found) in zip(bars, untyped):
        if not found:
            bar.set_hatch("//")
    axis.axhline(UNTYPED_BUDGET, linestyle=":", color="gray")
    axis.set_yscale("log")
    axis.set_xticks(list(positions))
    axis.set_xticklabels(labels)
    axis.set_ylabel("expansions (log)")
    axis.set_title("types as dimensional pruning (hatched = budget hit)")
    axis.legend()


def _plot_priors(axis, priors):
    bars = axis.bar([label for label, _, _ in priors],
                    [value for _, value, _ in priors])
    for bar, (_, _, found) in zip(bars, priors):
        if not found:
            bar.set_hatch("//")
    axis.axhline(PRIOR_BUDGET, linestyle=":", color="gray")
    axis.set_yscale("log")
    axis.set_ylabel("expansions (log)")
    axis.set_title("unigram priors on untyped potential-energy search")


def _plot_bloat(axis, bloat):
    counts, expansions = bloat
    axis.plot(counts, expansions, marker="o")
    axis.set_xlabel("distractor inventions")
    axis.set_ylabel("expansions")
    axis.set_title("library bloat: junk inventions slow everyone down")


def plot_results(
    tasks, frozen, growing, events, ablation, priors, bloat, log=lambda _: None
):
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))

    _plot_curriculum(axes[0][0], tasks, frozen, growing, events)
    _plot_ablation(axes[0][1], ablation)
    _plot_priors(axes[1][0], priors)
    _plot_bloat(axes[1][1], bloat)

    figure.suptitle("library learning over physics formulas")
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=150)
    log(f"\nwrote {PLOT_PATH}")


def main() -> None:
    tasks = curriculum_tasks()
    frozen, growing, inventions, events = run_curriculum(tasks, log=print)
    ablation = run_typing_ablation(tasks, frozen, log=print)
    priors = run_prior_comparison(tasks, log=print)
    bloat = run_library_bloat(inventions, tasks[-1], log=print)
    plot_results(tasks, frozen, growing, events, ablation, priors, bloat, log=print)


if __name__ == "__main__":
    main()
