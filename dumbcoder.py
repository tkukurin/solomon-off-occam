# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "pytorch>=1.0.2",
# ]
# ///
"""DreamCoder-style Explore, Compress, Dream (ECD) synthesis loop.

Provides the solver for programmatic synthesis given a Domain Specific Language.
"""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Delta:
    head: Any
    type: Any = None
    tailtypes: tuple[Any, ...] = ()
    tails: tuple['Delta', ...] = ()
    hiddentail: 'Delta' | None = None
    ishole: bool = False
    isarg: bool = False
    repr_str: str | None = None

    def __post_init__(self):
        if self.repr_str is None:
            r = str(self.head)
            if not self.ishole and not self.isarg and self.type is str:
                r = f"'{r}'"
            object.__setattr__(self, 'repr_str', r)

    def __call__(self):
        if not self.tails:
            return self.head
        if self.hiddentail:
            body = self.hiddentail
            for tidx, tail in enumerate(self.tails):
                body = replace_hidden(body, Delta(f'${tidx}', isarg=True, type=tail.type), tail)
            return body()

        args = [a() if isinstance(a, Delta) else a for a in self.tails]
        return self.head(*args)

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        if not isinstance(other, Delta): return False
        if self.ishole or other.ishole: return self.type == other.type
        if self.isarg and other.isarg: return self.type == other.type
        return self.head == other.head and self.tails == other.tails

    def __repr__(self):
        if not self.tails:
            return self.repr_str or str(self.head)
        return f'({self.repr_str} {" ".join(map(str, self.tails))})'


def isterminal(d: Delta) -> bool:
    if not d.tailtypes:
        return True
    if not d.tails:
        return False
    return all(isterminal(t) for t in d.tails)


def length(tree: Delta) -> int:
    if not tree: return 0
    if not tree.tails: return 1
    return 1 + sum(length(t) for t in tree.tails)


def countholes(tree: Delta) -> int:
    if not tree: return 0
    if tree.ishole: return 1
    if not tree.tails: return 0
    return sum(countholes(t) for t in tree.tails)


def extract_matches(tree: Delta, treeholed: Delta) -> list[tuple[Any, Delta]]:
    if not tree or not treeholed: return []
    if treeholed.ishole or treeholed.isarg:
        return [(treeholed.head, tree)]
    if not tree.tails: return []
    return [match for t, ht in zip(tree.tails, treeholed.tails) for match in extract_matches(t, ht)]


def replace_hidden(tree: Delta, arg: Delta, tail: Delta) -> Delta:
    if tree == arg: return tail
    if not tree.tails: return tree
    new_tails = tuple(replace_hidden(t, arg, tail) for t in tree.tails)
    return replace(tree, tails=new_tails)


def replace_tree(tree: Delta, matchbranch: Delta, newbranch: Delta) -> Delta:
    if tree == matchbranch:
        args = dict(extract_matches(tree, matchbranch))
        return replace(newbranch, tails=tuple(args.values())) if args else newbranch
    if not tree.tails: return tree
    return replace(tree, tails=tuple(replace_tree(t, matchbranch, newbranch) for t in tree.tails))


def typize(tree: Delta) -> tuple[tuple[Any, ...], Delta]:
    tailtypes = []
    z = [0]

    def _typize(n: Delta) -> Delta:
        if n.ishole:
            tailtypes.append(n.type)
            ret = Delta(f'${z[0]}', ishole=True, type=n.type)
            z[0] += 1
            return ret
        if not n.tails: return n
        return replace(n, tails=tuple(_typize(t) for t in n.tails))

    new_tree = _typize(tree)
    return tuple(tailtypes), new_tree


def showoff_types(tree: Delta) -> set[Any]:
    types = {tree.type}
    if tree.tails:
        for t in tree.tails:
            types |= showoff_types(t)
    return types


def freeze(tree: Delta) -> Delta:
    if tree.ishole: return replace(tree, ishole=False, isarg=True)
    if not tree.tails: return tree
    return replace(tree, tails=tuple(freeze(t) for t in tree.tails))


def normalize(tree: Delta) -> Delta:
    if tree.hiddentail:
        ht = normalize(tree.hiddentail)
        if tree.tails:
            for tidx, tail in enumerate(tree.tails):
                ht = replace_hidden(ht, Delta(f'${tidx}', isarg=True, type=tail.type), normalize(tail))
        return ht
    if not tree.tails: return tree
    return replace(tree, tails=tuple(normalize(t) for t in tree.tails))


class Deltas:
    def __init__(self, core: list[Delta]):
        self.core = core
        self.invented: list[Delta] = []
        self.reset()

    def add(self, d: Delta):
        self.invented.append(d)
        self.infer()

    def pop(self, d: Delta):
        self.invented.pop(self.index(d) - len(self.core))
        self.infer()

    def reset(self):
        self.invented = []
        self.infer()

    def infer(self):
        self.ds = self.core + self.invented
        self.bytype_terminal = defaultdict(list)
        self.bytype = defaultdict(list)
        for i, d in enumerate(self.ds):
            if not d.tailtypes:
                self.bytype_terminal[d.type].append(i)
            self.bytype[d.type].append(i)

    def index(self, d: Delta | str) -> int:
        if isinstance(d, str):
            for i, ds in enumerate(self.ds):
                if ds.head == d or ds.repr_str == d: return i
            raise ValueError(f"{d} not found in DSL")
        return self.ds.index(d)

    def __iter__(self):
        return iter(self.ds)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int | str):
        if isinstance(idx, str):
            return self.ds[self.index(idx)]
        return self.ds[idx]


def count_everyone(X: str) -> int:
    return sum(len(X) - i for i in range(len(X)))


def _find_hole(n: Delta) -> list[int] | None:
    if n.ishole: return []
    for i, t in enumerate(n.tails):
        p = _find_hole(t)
        if p is not None: return [i] + p
    return None


def _replace_at_path(tree: Delta, path: list[int], new_node: Delta) -> Delta:
    if not path:
        return new_node
    idx = path[0]
    new_tails = list(tree.tails)
    new_tails[idx] = _replace_at_path(new_tails[idx], path[1:], new_node)
    return replace(tree, tails=tuple(new_tails))


def solve_enumeration(
    X: str, D: Deltas, Q: torch.Tensor, solutions: dict | None = None,
    maxdepth: int = 10, timeout: int = 60, budget: int = 0
):
    solutions = solutions or {}
    stime = time.time()

    q = [(0.0, replace(Delta('root', ishole=True, type=type(X))), ())]

    cnt = 0
    while q and time.time() - stime < timeout:
        q.sort(key=lambda x: x[0])
        logp, tree, path = q.pop(0)

        hole_path = _find_hole(tree)
        if hole_path is None:
            cnt += 1
            _check_solution(tree, X, solutions)
            if X in solutions: break
            continue

        curr = tree
        for p in hole_path: curr = curr.tails[p]

        for i in D.bytype[curr.type]:
            p = logp - Q[i].item()
            if p > budget and budget > 0: continue

            d = D[i]
            if d.tailtypes:
                new_d = replace(d, tails=tuple(Delta('<>', ishole=True, type=t) for t in d.tailtypes))
            else:
                new_d = d

            new_tree = _replace_at_path(tree, hole_path, new_d)
            q.append((p, new_tree, path + (i,)))

    return solutions


def _check_solution(tree: Delta, X: str, solutions: dict):
    try:
        out = tree()
    except Exception:
        return

    if isinstance(out, str) and out in X:
        if out not in solutions or length(tree) < length(solutions[out]):
            solutions[out] = tree


def count_jive(trees: list[Delta]):
    counts = Counter()

    def walk(n: Delta):
        if n.ishole or n.isarg: return
        if n.tails:
            counts[replace(n, tails=tuple(Delta('<>', ishole=True, type=t.type) for t in n.tails))] += 1
            for t in n.tails: walk(t)

    for t in trees: walk(t)
    return counts


def saturate(D: Deltas, sols: dict):
    trees = [normalize(s) for s in sols.values() if s]
    D.reset()

    while True:
        counts = count_jive(trees)
        if not counts: return trees

        mx = sum(length(t) for t in trees)
        hiddentail = _find_best_compression(counts, mx)
        if hiddentail is None: return trees

        tailtypes, ht_typed = typize(hiddentail)
        name = f"f{len(D.invented)}"
        df = Delta(name, type=hiddentail.type, tailtypes=tailtypes, hiddentail=ht_typed, repr_str=name)

        trees = [replace_tree(t, hiddentail, df) for t in trees]
        trees = [freeze(t) for t in trees]
        df = freeze(df)
        D.add(df)


def _find_best_compression(counts: Counter, mx: int) -> Delta | None:
    mk = 0.99
    hiddentail = None

    for ghost, c in counts.items():
        if c < 2: continue
        nargs = 1 + countholes(ghost)
        mxj = mx - c * (length(ghost) - nargs)
        mj = length(ghost)
        k = (mxj + mj) / mx

        if k < mk:
            mk = k
            hiddentail = ghost

    return hiddentail


def ECD(X: str, D: Deltas, timeout: int = 60, budget: int = 20):
    D.reset()
    Q = F.log_softmax(torch.ones(len(D)), -1)

    sols = {}
    solved = False
    idx = 0

    while not solved and idx < 1:
        sols = solve_enumeration(X, D, Q, sols, maxdepth=10, timeout=timeout, budget=budget)
        saturate(D, sols)

        if X in sols:
            break
        idx += 1

    return sols
