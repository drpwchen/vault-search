"""Personalized PageRank over the wiki-link adjacency graph.

LinearRAG / HippoRAG style "entity activation -> global importance aggregation":
a random walk with restart seeded on one or more notes ranks every other note by
multi-hop graph proximity. Pure-Python, no LLM, fast (only touches reached nodes).

Validated on this vault: replacing naive 1-hop expansion / shared-link count with
PPR lifts sparse-bridge recall@10 from ~60% to ~84%.
"""
from __future__ import annotations
from collections import defaultdict


def personalized_pagerank(
    adjacency: dict[str, list[str]],
    seeds: list[str],
    damping: float = 0.85,
    iters: int = 30,
    top_k: int = 20,
    exclude: set[str] | None = None,
) -> list[tuple[str, float]]:
    """Rank notes by PPR proximity to the seed notes.

    Args:
        adjacency: note -> list of linked note names (wiki-link graph).
        seeds: restart set (e.g. a single note for "related", or the top
            semantic-search hits for query expansion).
        top_k: number of ranked notes to return.
        exclude: notes to drop from the result (defaults to the seed set).

    Returns:
        [(note, score), ...] sorted by score desc, seeds/excluded removed.
    """
    seeds = [s for s in seeds if s]
    if not seeds:
        return []
    restart = {s: 1.0 / len(seeds) for s in seeds}
    score: dict[str, float] = dict(restart)
    for _ in range(iters):
        nxt: dict[str, float] = defaultdict(float)
        for node, s in score.items():
            nbrs = adjacency.get(node)
            if nbrs:
                w = damping * s / len(nbrs)
                for nb in nbrs:
                    nxt[nb] += w
        for s_node, rv in restart.items():
            nxt[s_node] += (1.0 - damping) * rv
        score = nxt
    drop = set(exclude) if exclude is not None else set(seeds)
    ranked = sorted(
        ((n, sc) for n, sc in score.items() if n not in drop),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return ranked[:top_k]


def related_notes(adjacency: dict[str, list[str]], note: str, top_k: int = 20) -> list[tuple[str, float]]:
    """Convenience: notes most related to a single note (restart on that note)."""
    return personalized_pagerank(adjacency, [note], top_k=top_k)
