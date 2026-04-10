#!/usr/bin/env python3
"""
Fork LongMemEval Benchmark — tests our provider/format/reranker stack.

Thin wrapper that reuses the upstream benchmark's data format and metrics
but injects our pluggable providers (oMLX embedding, cross-encoder rerank)
and compression formats (AAAK, Wenjian) into the evaluation loop.

Usage:
    # Baseline: ChromaDB default embedding, raw text
    python benchmarks/fork_longmemeval.py benchmarks/data/longmemeval_s_cleaned.json --mode raw

    # oMLX embedding
    python benchmarks/fork_longmemeval.py benchmarks/data/longmemeval_s_cleaned.json --mode raw --embedder omlx

    # oMLX embedding + cross-encoder rerank
    python benchmarks/fork_longmemeval.py benchmarks/data/longmemeval_s_cleaned.json --mode raw --embedder omlx --rerank

    # AAAK compression + oMLX
    python benchmarks/fork_longmemeval.py benchmarks/data/longmemeval_s_cleaned.json --mode aaak --embedder omlx

    # Wenjian compression + oMLX + rerank
    python benchmarks/fork_longmemeval.py benchmarks/data/longmemeval_s_cleaned.json --mode wenjian --embedder omlx --rerank
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb


# =============================================================================
# METRICS (from upstream longmemeval_bench.py)
# =============================================================================

def dcg(relevances, k):
    score = 0.0
    for i, rel in enumerate(relevances[:k]):
        score += rel / math.log2(i + 2)
    return score


def ndcg(rankings, correct_ids, corpus_ids, k):
    relevances = [1.0 if corpus_ids[idx] in correct_ids else 0.0 for idx in rankings[:k]]
    ideal = sorted(relevances, reverse=True)
    idcg = dcg(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg(relevances, k) / idcg


def evaluate_retrieval(rankings, correct_ids, corpus_ids, k):
    top_k_ids = set(corpus_ids[idx] for idx in rankings[:k])
    recall_any = float(any(cid in top_k_ids for cid in correct_ids))
    ndcg_score = ndcg(rankings, correct_ids, corpus_ids, k)
    return recall_any, ndcg_score


def session_id_from_corpus_id(corpus_id):
    if "_turn_" in corpus_id:
        return corpus_id.rsplit("_turn_", 1)[0]
    return corpus_id


# =============================================================================
# EMBEDDER SETUP
# =============================================================================

_client = chromadb.EphemeralClient()
_embed_fn = None
_reranker = None


def setup_embedder(embedder_name: str, omlx_url: str):
    global _embed_fn
    if embedder_name == "default":
        _embed_fn = None
        return

    if embedder_name == "omlx":
        from mempalace.providers.openai_compatible import OpenAICompatibleClient
        from mempalace.providers.chroma_adapter import ChromaProviderAdapter

        omlx = OpenAICompatibleClient(
            base_url=omlx_url,
            embed_model="Qwen3-Embedding-0.6B-8bit",
            embed_batch_size=48,
            timeout_seconds=30.0,
        )
        _embed_fn = ChromaProviderAdapter(omlx)
        # Warmup to verify connectivity
        test_vec = omlx.embed("warmup")
        print(f"  Embedder: oMLX Qwen3-Embedding-0.6B-8bit (dim={len(test_vec)})")
        return

    raise ValueError(f"Unknown embedder: {embedder_name}")


def setup_reranker(omlx_url: str):
    global _reranker
    from mempalace.providers.openai_compatible import OpenAICompatibleClient
    from mempalace.rerankers import Reranker

    omlx = OpenAICompatibleClient(
        base_url=omlx_url,
        rerank_model="Qwen3-Reranker-4B-4bit-MLX",
        timeout_seconds=120.0,  # reranking 50+ docs takes ~8s per question
    )
    _reranker = Reranker(omlx, candidate_multiplier=1)  # we control fetch size ourselves
    print(f"  Reranker: oMLX Qwen3-Reranker-4B-4bit-MLX")


def fresh_collection(name="bench_drawers"):
    global _embed_fn
    try:
        _client.delete_collection(name)
    except Exception:
        pass
    if _embed_fn is not None:
        return _client.create_collection(name, embedding_function=_embed_fn, metadata={"hnsw:space": "cosine"})
    return _client.create_collection(name)


# =============================================================================
# COMPRESSION
# =============================================================================

_compressor = None


def setup_compressor(mode: str):
    global _compressor
    if mode == "raw":
        _compressor = None
        return

    if mode == "aaak":
        from mempalace.dialect import Dialect
        _compressor = ("aaak", Dialect())
        return

    if mode == "wenjian":
        from mempalace.formats.wenjian import WenjianFormat
        _compressor = ("wenjian", WenjianFormat())
        return

    raise ValueError(f"Unknown mode: {mode}")


def compress_text(text: str) -> str:
    if _compressor is None:
        return text
    name, compressor = _compressor
    if name == "aaak":
        return compressor.compress(text)
    if name == "wenjian":
        entry = compressor.compress(text)
        return entry.text
    return text


# =============================================================================
# RETRIEVAL
# =============================================================================

def run_single_question(entry, n_results=50):
    """Ingest sessions, query, optionally rerank. Returns ranked indices."""
    corpus = []
    corpus_ids = []

    for session, sess_id in zip(entry["haystack_sessions"], entry["haystack_session_ids"]):
        user_turns = [t["content"] for t in session if t["role"] == "user"]
        if user_turns:
            doc = "\n".join(user_turns)
            corpus.append(doc)
            corpus_ids.append(sess_id)

    if not corpus:
        return [], corpus, corpus_ids

    # Compress if needed
    ingestion_docs = [compress_text(doc) for doc in corpus]

    col = fresh_collection()
    col.add(
        documents=ingestion_docs,
        ids=[f"doc_{i}" for i in range(len(ingestion_docs))],
    )

    # Query (always with raw question, not compressed)
    query = entry["question"]
    fetch_n = min(n_results, len(corpus))
    results = col.query(
        query_texts=[query],
        n_results=fetch_n,
        include=["distances", "documents"],
    )

    result_ids = results["ids"][0]
    doc_id_to_idx = {f"doc_{i}": i for i in range(len(corpus))}

    if _reranker is not None:
        # Build candidate list for reranker using ORIGINAL text (not compressed)
        candidates = []
        for rid in result_ids:
            idx = doc_id_to_idx[rid]
            candidates.append({"text": corpus[idx], "idx": idx})

        reranked = _reranker.rerank(query, candidates, top_n=fetch_n)
        ranked_indices = [c["idx"] for c in reranked]
    else:
        ranked_indices = [doc_id_to_idx[rid] for rid in result_ids]

    # Fill remaining indices
    seen = set(ranked_indices)
    for i in range(len(corpus)):
        if i not in seen:
            ranked_indices.append(i)

    return ranked_indices, corpus, corpus_ids


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fork LongMemEval Benchmark")
    parser.add_argument("data_file", help="Path to longmemeval_s_cleaned.json")
    parser.add_argument("--mode", choices=["raw", "aaak", "wenjian"], default="raw")
    parser.add_argument("--embedder", choices=["default", "omlx"], default="default")
    parser.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranking")
    parser.add_argument("--omlx-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--limit", type=int, default=0, help="Limit questions (0=all)")
    parser.add_argument("--k", type=int, default=5, help="Recall@k (default 5)")
    parser.add_argument("--output", default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    # Load data
    print(f"\n  Loading {args.data_file}...")
    with open(args.data_file) as f:
        data = json.load(f)
    if args.limit > 0:
        data = data[:args.limit]
    print(f"  Questions: {len(data)}")

    # Setup
    config_label = f"mode={args.mode} embedder={args.embedder}"
    if args.rerank:
        config_label += " +rerank"
    print(f"  Config: {config_label}")

    setup_embedder(args.embedder, args.omlx_url)
    setup_compressor(args.mode)
    if args.rerank:
        setup_reranker(args.omlx_url)

    # Run
    print(f"\n  Running {len(data)} questions (k={args.k})...\n")
    k = args.k
    results_by_type = defaultdict(list)
    all_recalls = []
    errors = 0
    t0 = time.time()

    for i, entry in enumerate(data):
        qtype = entry.get("question_type", "unknown")
        correct_ids = set(entry["answer_session_ids"])

        try:
            rankings, corpus, corpus_ids = run_single_question(entry)
        except Exception as e:
            print(f"  ERROR q{i}: {e}")
            errors += 1
            continue

        if not corpus:
            continue

        recall, ndcg_score = evaluate_retrieval(rankings, correct_ids, corpus_ids, k)
        all_recalls.append(recall)
        results_by_type[qtype].append(recall)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            running_r = sum(all_recalls) / len(all_recalls) * 100
            print(f"  [{i+1:4d}/{len(data)}] R@{k}={running_r:.1f}% ({elapsed:.0f}s)")

    elapsed = time.time() - t0

    # Summary
    total_r = sum(all_recalls) / len(all_recalls) * 100 if all_recalls else 0
    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {config_label}")
    print(f"{'=' * 60}")
    print(f"  R@{k}: {total_r:.1f}% ({sum(int(r) for r in all_recalls)}/{len(all_recalls)})")
    print(f"  Time: {elapsed:.1f}s ({elapsed/len(data):.2f}s/question)")
    if errors:
        print(f"  Errors: {errors}")

    print(f"\n  Per question type:")
    for qtype in sorted(results_by_type.keys()):
        recalls = results_by_type[qtype]
        r = sum(recalls) / len(recalls) * 100
        print(f"    {qtype:25s}  R@{k}={r:.1f}% ({sum(int(x) for x in recalls)}/{len(recalls)})")

    print(f"\n{'=' * 60}\n")

    # Save results
    output = {
        "config": {
            "mode": args.mode,
            "embedder": args.embedder,
            "rerank": args.rerank,
            "k": k,
            "questions": len(data),
        },
        "recall_at_k": round(total_r, 2),
        "per_type": {
            qt: round(sum(rs) / len(rs) * 100, 2)
            for qt, rs in results_by_type.items()
        },
        "elapsed_seconds": round(elapsed, 1),
        "errors": errors,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Results saved to {args.output}")

    return output


if __name__ == "__main__":
    main()
