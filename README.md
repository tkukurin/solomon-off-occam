# solom-on-off-occam

priors, kolmogorov complexity, all that jazz.

## solomonoff

`mini_v0.py`: tiny end-to-end bridge from a computable solomonoff mixture to a PFN that amortizes bayes, then shows why shrinking to a domain prior helps in-domain.

`mini_v1.py`: same coin-world but stress tests. capacity vs sufficient statistics, latent domains, fine-tuning/forgetting, bottlenecks/LoRA, probes, raw-sequence attention.


some references
* [gdm-univ]: _amortizing the most powerful universal predictor, Solomonoff Induction (SI), into neural networks via meta-learning_ (...) _generate training data via UTMs to expose networks to a broad range of patterns_
* [gdm-metalrn]: _memory-based meta-learning as a tool for building sample-efficient strategies that learn from past experience to adapt to any task within a target class_
* [grunwald-mdl]: 1st chapter non-technical introduction to the subject, then technical in ch02


gemini expands wtih:
* [weight-norm-kc]: _proves weight decay in fixed-precision networks matches Solomonoff's universal prior; smallest weight norm equals KC up to a log factor_
* [wilson-dl]: _formalizes model compressibility via KC; shows large models fitting data well can be compressed to small filesizes, bounding K(h)_
* [mosaic-motifs]: _restricts weights to reusable motifs (MoMos) to demonstrably lower algorithmic complexity during training_


## dreamcoder

`dumbcoder.py` + `dumbcoder_physics.py`: tiny dreamcoder-ish program synthesis experiments—search, compress recurring programs into new primitives, repeat.

solomonoff-wise, learning the DSL means learning the coding scheme behind the prior: reusable structure gets shorter descriptions and more probability.

* [dreamcoder]: _wake-sleep Bayesian program learning; grows a DSL of reusable abstractions alongside a neural search policy_
* [stitch]: _compresses program corpora into reusable abstractions, much faster than dreamcoder's deductive library learning_
* [lilo]: _iterates synthesis, compression, and documentation; dreamcoder-ish library learning with LLM-guided search_


### domains

`uv run --extra minihack dumbcoder_minihack.py`: opens the MiniHack workbench immediately from cache, with live phase status on a cache miss. Use `--force` to recompute program learning, held-out ablations, controlled dynamics, time-expanded A*, and the potion/levitation/lava FSM.

`uv run dumbcoder_arc.py`: [ARC-AGI-1][arc-agi] program induction over a grid-typed slice of [arc-dsl] primitives. best-first enumeration (uniform prior, smallest-first) induces programs from one task's train pairs, then scores them on the held-out test pair; also surfaces semantic duplicates like `(hmirror (vconcat (hmirror $I) $I))`. task jsons cached under `data/arc/`.

todo: compression phase; mirror-tile tasks (e.g. `3af2c5a8`) need 10-node trees naively—the shared half appears twice without let-bindings—but collapse to 3-node compositions once `hconcat(x, vmirror(x))`-style inventions exist.



[gdm-univ]: <https://arxiv.org/abs/2401.14953> "Learning Universal Predictors"
[gdm-metalrn]: <https://arxiv.org/abs/1905.03030> "Meta-learning of Sequential Strategies"
[prior-v0]: <https://arxiv.org/abs/2112.10510> "Transformers Can Do Bayesian Inference"
[schmidhuber-speed-prior]: <https://gwern.net/doc/reinforcement-learning/model/2002-schmidhuber.pdf> "Speed Prior"
[grunwald-mdl]: <https://arxiv.org/pdf/math/0406077> "A Tutorial Introduction to the Minimum Description Length Principle"
[dreamcoder]: <https://arxiv.org/abs/2006.08381> "DreamCoder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning"
[stitch]: <https://arxiv.org/abs/2211.16605> "Top-Down Synthesis for Library Learning"
[lilo]: <https://arxiv.org/abs/2310.19791> "LILO: Learning Interpretable Libraries by Compressing and Documenting Code"
[arc-agi]: <https://arxiv.org/abs/1911.01547> "On the Measure of Intelligence"
[arc-dsl]: <https://github.com/michaelhodel/arc-dsl> "Domain Specific Language for the Abstraction and Reasoning Corpus"

[weight-norm-kc]: <https://arxiv.org/abs/2605.10878> "Neural Weight Norm = Kolmogorov Complexity"
[wilson-dl]: <https://arxiv.org/abs/2503.02113> "Deep Learning is Not So Mysterious or Different"
[mosaic-motifs]: <https://arxiv.org/abs/2602.14896> "Algorithmic Simplification of Neural Networks with Mosaic-of-Motifs"
[catastrophic-per-layer]: https://arxiv.org/abs/2007.07400 "Anatomy of Catastrophic Forgetting: Hidden Representations and Task Semantics"
