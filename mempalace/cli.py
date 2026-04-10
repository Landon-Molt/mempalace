#!/usr/bin/env python3
"""
MemPalace — Give your AI a memory. No API key required.

Two ways to ingest:
  Projects:      mempalace mine ~/projects/my_app          (code, docs, notes)
  Conversations: mempalace mine ~/chats/ --mode convos     (Claude, ChatGPT, Slack)

Same palace. Same search. Different ingest strategies.

Commands:
    mempalace init <dir>                  Detect rooms from folder structure
    mempalace split <dir>                 Split concatenated mega-files into per-session files
    mempalace mine <dir>                  Mine project files (default)
    mempalace mine <dir> --mode convos    Mine conversation exports
    mempalace search "query"              Find anything, exact words
    mempalace mcp                         Show MCP setup command
    mempalace wake-up                     Show L0 + L1 wake-up context
    mempalace wake-up --wing my_app       Wake-up for a specific project
    mempalace status                      Show what's been filed

Examples:
    mempalace init ~/projects/my_app
    mempalace mine ~/projects/my_app
    mempalace mine ~/chats/claude-sessions --mode convos
    mempalace search "why did we switch to GraphQL"
    mempalace search "pricing discussion" --wing my_app --room costs
"""

import os
import sys
import shlex
import argparse
from pathlib import Path

from .config import MempalaceConfig


def cmd_init(args):
    import json
    from pathlib import Path
    from .entity_detector import scan_for_detection, detect_entities, confirm_entities
    from .room_detector_local import detect_rooms_local

    # Pass 1: auto-detect people and projects from file content
    print(f"\n  Scanning for entities in: {args.dir}")
    files = scan_for_detection(args.dir)
    if files:
        print(f"  Reading {len(files)} files...")
        detected = detect_entities(files)
        total = len(detected["people"]) + len(detected["projects"]) + len(detected["uncertain"])
        if total > 0:
            confirmed = confirm_entities(detected, yes=getattr(args, "yes", False))
            # Save confirmed entities to <project>/entities.json for the miner
            if confirmed["people"] or confirmed["projects"]:
                entities_path = Path(args.dir).expanduser().resolve() / "entities.json"
                with open(entities_path, "w") as f:
                    json.dump(confirmed, f, indent=2)
                print(f"  Entities saved: {entities_path}")
        else:
            print("  No entities detected — proceeding with directory-based rooms.")

    # Pass 2: detect rooms from folder structure
    detect_rooms_local(project_dir=args.dir, yes=getattr(args, "yes", False))
    MempalaceConfig().init()


def cmd_mine(args):
    palace_path = os.path.expanduser(args.palace) if args.palace else MempalaceConfig().palace_path
    include_ignored = []
    for raw in args.include_ignored or []:
        include_ignored.extend(part.strip() for part in raw.split(",") if part.strip())

    if args.mode == "convos":
        from .convo_miner import mine_convos

        mine_convos(
            convo_dir=args.dir,
            palace_path=palace_path,
            wing=args.wing,
            agent=args.agent,
            limit=args.limit,
            dry_run=args.dry_run,
            extract_mode=args.extract,
        )
    else:
        from .miner import mine

        mine(
            project_dir=args.dir,
            palace_path=palace_path,
            wing_override=args.wing,
            agent=args.agent,
            limit=args.limit,
            dry_run=args.dry_run,
            respect_gitignore=not args.no_gitignore,
            include_ignored=include_ignored,
        )


def cmd_search(args):
    from .searcher import search, search_memories, SearchError
    from .providers import DimensionMismatchError

    palace_config = MempalaceConfig()
    palace_path = os.path.expanduser(args.palace) if args.palace else palace_config.palace_path

    # Determine rerank: --rerank flag, --no-rerank flag, or config default.
    rerank_flag = getattr(args, "rerank", None)

    try:
        if rerank_flag is not None:
            # Use the programmatic path which supports reranking.
            result = search_memories(
                query=args.query,
                palace_path=palace_path,
                wing=args.wing,
                room=args.room,
                n_results=args.results,
                config=palace_config,
                rerank=rerank_flag,
            )
            if "error" in result:
                print(f"\n  {result['error']}")
                if "hint" in result:
                    print(f"  {result['hint']}")
                sys.exit(1)

            # Print results in the same format as the print-based search.
            hits = result.get("results", [])
            reranked = result.get("reranked", False)
            if not hits:
                print(f'\n  No results found for: "{args.query}"')
                return

            print(f"\n{'=' * 60}")
            print(f'  Results for: "{args.query}"')
            if args.wing:
                print(f"  Wing: {args.wing}")
            if args.room:
                print(f"  Room: {args.room}")
            if reranked:
                print("  Reranked: yes")
            print(f"{'=' * 60}\n")

            for i, hit in enumerate(hits, 1):
                wing_name = hit.get("wing", "?")
                room_name = hit.get("room", "?")
                source = hit.get("source_file", "?")
                score = hit.get("rerank_score", hit.get("similarity", 0))
                print(f"  [{i}] {wing_name} / {room_name}")
                print(f"      Source: {source}")
                print(f"      Score:  {score:.3f}" + (" (reranked)" if "rerank_score" in hit else ""))
                print()
                for line in hit.get("text", "").strip().split("\n"):
                    print(f"      {line}")
                print()
                print(f"  {'─' * 56}")
            print()
        else:
            # Default: use the print-based search (no reranking).
            search(
                query=args.query,
                palace_path=palace_path,
                wing=args.wing,
                room=args.room,
                n_results=args.results,
                config=palace_config,
            )
    except DimensionMismatchError as e:
        print(f"\n  Dimension mismatch:\n{e}", file=sys.stderr)
        sys.exit(2)
    except SearchError:
        sys.exit(1)


def cmd_wakeup(args):
    """Show L0 (identity) + L1 (essential story) — the wake-up context."""
    from .layers import MemoryStack

    palace_path = os.path.expanduser(args.palace) if args.palace else MempalaceConfig().palace_path
    stack = MemoryStack(palace_path=palace_path)

    text = stack.wake_up(wing=args.wing)
    tokens = len(text) // 4
    print(f"Wake-up text (~{tokens} tokens):")
    print("=" * 50)
    print(text)


def cmd_split(args):
    """Split concatenated transcript mega-files into per-session files."""
    from .split_mega_files import main as split_main
    import sys

    # Rebuild argv for split_mega_files argparse
    argv = ["--source", args.dir]
    if args.output_dir:
        argv += ["--output-dir", args.output_dir]
    if args.dry_run:
        argv.append("--dry-run")
    if args.min_sessions != 2:
        argv += ["--min-sessions", str(args.min_sessions)]

    old_argv = sys.argv
    sys.argv = ["mempalace split"] + argv
    try:
        split_main()
    finally:
        sys.argv = old_argv


def cmd_status(args):
    from .miner import status

    palace_path = os.path.expanduser(args.palace) if args.palace else MempalaceConfig().palace_path
    status(palace_path=palace_path)


def cmd_repair(args):
    """Rebuild palace vector index from SQLite metadata."""
    import shutil

    import chromadb

    from .palace import get_collection

    palace_config = MempalaceConfig()
    palace_path = (
        os.path.expanduser(args.palace) if args.palace else palace_config.palace_path
    )

    if not os.path.isdir(palace_path):
        print(f"\n  No palace found at {palace_path}")
        return

    print(f"\n{'=' * 55}")
    print("  MemPalace Repair")
    print(f"{'=' * 55}\n")
    print(f"  Palace: {palace_path}")

    # Try to read existing drawers through the configured embedder.
    try:
        col = get_collection(palace_path, config=palace_config)
        total = col.count()
        print(f"  Drawers found: {total}")
    except Exception as e:
        print(f"  Error reading palace: {e}")
        print("  Cannot recover — palace may need to be re-mined from source files.")
        return

    if total == 0:
        print("  Nothing to repair.")
        return

    # Extract all drawers in batches
    print("\n  Extracting drawers...")
    batch_size = 5000
    all_ids = []
    all_docs = []
    all_metas = []
    offset = 0
    while offset < total:
        batch = col.get(limit=batch_size, offset=offset, include=["documents", "metadatas"])
        all_ids.extend(batch["ids"])
        all_docs.extend(batch["documents"])
        all_metas.extend(batch["metadatas"])
        offset += batch_size
    print(f"  Extracted {len(all_ids)} drawers")

    # Backup and rebuild
    palace_path = palace_path.rstrip(os.sep)
    backup_path = palace_path + ".backup"
    if os.path.exists(backup_path):
        shutil.rmtree(backup_path)
    print(f"  Backing up to {backup_path}...")
    shutil.copytree(palace_path, backup_path)

    print("  Rebuilding collection...")
    # Collection-level maintenance op — delete then recreate. We need a raw
    # client for delete_collection; the subsequent get_collection call
    # recreates the collection with the correct config-resolved embedder.
    raw_client = chromadb.PersistentClient(path=palace_path)
    raw_client.delete_collection("mempalace_drawers")
    new_col = get_collection(palace_path, config=palace_config)

    filed = 0
    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i : i + batch_size]
        batch_docs = all_docs[i : i + batch_size]
        batch_metas = all_metas[i : i + batch_size]
        new_col.add(documents=batch_docs, ids=batch_ids, metadatas=batch_metas)
        filed += len(batch_ids)
        print(f"  Re-filed {filed}/{len(all_ids)} drawers...")

    print(f"\n  Repair complete. {filed} drawers rebuilt.")
    print(f"  Backup saved at {backup_path}")
    print(f"\n{'=' * 55}\n")


def cmd_hook(args):
    """Run hook logic: reads JSON from stdin, outputs JSON to stdout."""
    from .hooks_cli import run_hook

    run_hook(hook_name=args.hook, harness=args.harness)


def cmd_instructions(args):
    """Output skill instructions to stdout."""
    from .instructions_cli import run_instructions

    run_instructions(name=args.name)


def cmd_mcp(args):
    """Show how to wire MemPalace into MCP-capable hosts."""
    base_server_cmd = "python -m mempalace.mcp_server"

    if args.palace:
        resolved_palace = str(Path(args.palace).expanduser())
        server_cmd = f"{base_server_cmd} --palace {shlex.quote(resolved_palace)}"
    else:
        server_cmd = base_server_cmd

    print("MemPalace MCP quick setup:")
    print(f"  claude mcp add mempalace -- {server_cmd}")
    print("\nRun the server directly:")
    print(f"  {server_cmd}")

    if not args.palace:
        print("\nOptional custom palace:")
        print(f"  claude mcp add mempalace -- {base_server_cmd} --palace /path/to/palace")
        print(f"  {base_server_cmd} --palace /path/to/palace")


def cmd_compress(args):
    """Compress drawers using AAAK or Wenjian format."""
    from .formats import resolve_format
    from .palace import get_collection
    from .providers import resolve_llm
    from .summarizer import Summarizer

    palace_config = MempalaceConfig()
    palace_path = (
        os.path.expanduser(args.palace) if args.palace else palace_config.palace_path
    )

    # Determine format: --format flag > config > default (aaak for backward compat)
    compression_cfg = palace_config.compression
    if not isinstance(compression_cfg, dict):
        compression_cfg = {}
    format_name = (
        getattr(args, "format", None)
        or compression_cfg.get("format")
        or "aaak"
    )
    rule_only = (
        getattr(args, "rule_only", False)
        or compression_cfg.get("rule_only", False)
    )

    # Build the summarizer with format + optional LLM.
    fmt_kwargs = {}
    if format_name == "aaak":
        # Entity config for AAAK
        config_path = args.config
        if not config_path:
            for candidate in ["entities.json", os.path.join(palace_path, "entities.json")]:
                if os.path.exists(candidate):
                    config_path = candidate
                    break
        if config_path and os.path.exists(config_path):
            fmt_kwargs["config_path"] = config_path
            print(f"  Loaded entity config: {config_path}")

    fmt = resolve_format(format_name, **fmt_kwargs)
    llm = None if rule_only else resolve_llm(palace_config)
    summarizer = Summarizer(fmt, llm=llm, rule_only=rule_only)

    mode_label = f"{format_name}"
    if summarizer.has_llm:
        mode_label += " + LLM"
    else:
        mode_label += " (rule-only)"
    print(f"  Format: {mode_label}")

    # Connect to palace
    try:
        col = get_collection(palace_path, config=palace_config)
    except Exception:
        print(f"\n  No palace found at {palace_path}")
        print("  Run: mempalace init <dir> then mempalace mine <dir>")
        sys.exit(1)

    # Query drawers in batches to avoid SQLite variable limit (~999)
    where = {"wing": args.wing} if args.wing else None
    _BATCH = 500
    docs, metas, ids = [], [], []
    offset = 0
    while True:
        try:
            kwargs = {"include": ["documents", "metadatas"], "limit": _BATCH, "offset": offset}
            if where:
                kwargs["where"] = where
            batch = col.get(**kwargs)
        except Exception as e:
            if not docs:
                print(f"\n  Error reading drawers: {e}")
                sys.exit(1)
            break
        batch_docs = batch.get("documents", [])
        if not batch_docs:
            break
        docs.extend(batch_docs)
        metas.extend(batch.get("metadatas", []))
        ids.extend(batch.get("ids", []))
        offset += len(batch_docs)
        if len(batch_docs) < _BATCH:
            break

    if not docs:
        wing_label = f" in wing '{args.wing}'" if args.wing else ""
        print(f"\n  No drawers found{wing_label}.")
        return

    print(
        f"\n  Compressing {len(docs)} drawers"
        + (f" in wing '{args.wing}'" if args.wing else "")
        + "..."
    )
    print()

    total_original = 0
    total_compressed = 0
    compressed_entries = []

    for doc, meta, doc_id in zip(docs, metas, ids):
        entry = summarizer.compress(doc, context=meta)

        total_original += len(doc)
        total_compressed += len(entry.text)

        compressed_entries.append((doc_id, entry, meta))

        if args.dry_run:
            wing_name = meta.get("wing", "?")
            room_name = meta.get("room", "?")
            source = Path(meta.get("source_file", "?")).name
            print(f"  [{wing_name}/{room_name}] {source}")
            ratio = entry.compression_ratio
            print(
                f"    {entry.original_token_count}t -> "
                f"{entry.compressed_token_count}t "
                f"({ratio:.1f}x) "
                f"pres={entry.entities_preserved_ratio:.0%}"
            )
            print(f"    {entry.text[:200]}")
            if entry.missing_entities:
                print(f"    missing: {', '.join(entry.missing_entities[:5])}")
            print()

    # Store compressed versions (unless dry-run)
    if not args.dry_run:
        try:
            comp_col = get_collection(
                palace_path,
                collection_name="mempalace_compressed",
                config=palace_config,
            )
            for doc_id, entry, meta in compressed_entries:
                comp_meta = dict(meta)
                comp_meta["compression_ratio"] = round(entry.compression_ratio, 1)
                comp_meta["original_tokens"] = entry.original_token_count
                comp_meta["format_name"] = entry.format_name
                comp_meta["preservation_ratio"] = round(entry.entities_preserved_ratio, 2)
                if entry.llm_model:
                    comp_meta["llm_model"] = entry.llm_model
                comp_col.upsert(
                    ids=[doc_id],
                    documents=[entry.text],
                    metadatas=[comp_meta],
                )
            print(
                f"  Stored {len(compressed_entries)} compressed drawers in 'mempalace_compressed' collection."
            )
        except Exception as e:
            print(f"  Error storing compressed drawers: {e}")
            sys.exit(1)

    # Summary
    ratio = total_original / max(total_compressed, 1)
    print(f"\n  Total: {len(compressed_entries)} drawers compressed ({ratio:.1f}x avg)")
    print(f"  Format: {format_name}")
    if args.dry_run:
        print("  (dry run -- nothing stored)")


def cmd_reembed(args):
    """Re-embed all drawers with the current config-resolved provider.

    Used when the user switches embedding models or changes backend
    (e.g., from ChromaDB default 384-dim to oMLX Qwen3 1024-dim). Walks
    both the main ``mempalace_drawers`` collection and
    ``mempalace_compressed`` if it exists, re-adds every document through
    the new embedder, and updates the palace sidecar.

    Idempotent: if the sidecar already matches the current config,
    prints "up to date" and exits. Never destroys the palace directory —
    takes a full backup under ``{palace}.backup`` first.
    """
    import shutil

    import chromadb

    from .palace import get_collection
    from .palace_meta import read_meta, write_meta, meta_from_embedder, META_FILENAME
    from .providers import ProviderError, resolve_embedder

    palace_config = MempalaceConfig()
    palace_path = (
        os.path.expanduser(args.palace) if args.palace else palace_config.palace_path
    )

    if not os.path.isdir(palace_path):
        print(f"\n  No palace found at {palace_path}")
        sys.exit(1)

    target_embedder = resolve_embedder(palace_config)
    if target_embedder is None:
        print(
            "\n  No embedding provider configured — cannot reembed.\n"
            "  Set `embedding` in ~/.mempalace/config.json or use env vars "
            "(MEMPALACE_EMBED_PROVIDER, MEMPALACE_EMBED_MODEL, MEMPALACE_EMBED_URL).\n"
        )
        sys.exit(1)

    # Warmup so we know the target dimension before we touch anything.
    if target_embedder.embed_dim is None:
        try:
            target_embedder.embed("mempalace reembed warmup")
        except Exception as e:
            print(f"\n  Target embedder is unreachable: {e}")
            sys.exit(1)

    current_dim = target_embedder.embed_dim
    current_model = target_embedder.model_name

    stored_meta = read_meta(palace_path)

    print(f"\n{'=' * 55}")
    print("  MemPalace Reembed")
    print(f"{'=' * 55}")
    print(f"  Palace: {palace_path}")
    if stored_meta:
        print(
            f"  From:   {stored_meta.embedding_model} "
            f"(dim={stored_meta.embedding_dim}, provider={stored_meta.embedding_provider})"
        )
    else:
        print("  From:   <unknown — no sidecar>")
    print(f"  To:     {current_model} (dim={current_dim})")

    if (
        stored_meta
        and stored_meta.embedding_dim == current_dim
        and stored_meta.embedding_model == current_model
    ):
        print("\n  Palace is already embedded with the current provider. Nothing to do.")
        return

    # Read the existing drawers using a bypass-enabled get_collection so the
    # dim check doesn't block us. .get() doesn't hit the embedder at all —
    # it's a metadata+document query.
    try:
        old_col = get_collection(
            palace_path,
            collection_name="mempalace_drawers",
            config=palace_config,
            allow_dim_mismatch=True,
        )
        total = old_col.count()
    except Exception as e:
        print(f"\n  Error opening current palace: {e}")
        sys.exit(1)

    if total == 0:
        print("\n  Palace has no drawers. Updating sidecar only.")
        new_meta = meta_from_embedder(target_embedder)
        if stored_meta is not None:
            new_meta.created_at = stored_meta.created_at
        write_meta(palace_path, new_meta)
        return

    if not args.force:
        try:
            response = input(f"\n  Re-embed {total} drawers? [y/N] ").strip().lower()
        except EOFError:
            response = ""
        if response != "y":
            print("  Aborted.")
            return

    # Backup first. Same pattern as cmd_repair.
    palace_path_norm = palace_path.rstrip(os.sep)
    backup_path = palace_path_norm + ".backup"
    if os.path.exists(backup_path):
        shutil.rmtree(backup_path)
    print(f"\n  Backing up to {backup_path}...")
    shutil.copytree(palace_path_norm, backup_path)

    # Drain drawers.
    print(f"  Extracting {total} drawers...")
    batch_size = 500
    all_ids, all_docs, all_metas = [], [], []
    offset = 0
    while offset < total:
        batch = old_col.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )
        all_ids.extend(batch.get("ids", []))
        all_docs.extend(batch.get("documents", []))
        all_metas.extend(batch.get("metadatas", []))
        offset += batch_size

    # Drain compressed collection if present.
    comp_ids, comp_docs, comp_metas = [], [], []
    try:
        comp_col = get_collection(
            palace_path,
            collection_name="mempalace_compressed",
            config=palace_config,
            allow_dim_mismatch=True,
        )
        comp_total = comp_col.count()
        if comp_total > 0:
            print(f"  Extracting {comp_total} compressed drawers...")
            offset = 0
            while offset < comp_total:
                batch = comp_col.get(
                    limit=batch_size,
                    offset=offset,
                    include=["documents", "metadatas"],
                )
                comp_ids.extend(batch.get("ids", []))
                comp_docs.extend(batch.get("documents", []))
                comp_metas.extend(batch.get("metadatas", []))
                offset += batch_size
    except Exception:
        pass

    # Delete old collections (maintenance op — raw client).
    print("  Dropping old collections...")
    raw_client = chromadb.PersistentClient(path=palace_path)
    for name in ("mempalace_drawers", "mempalace_compressed"):
        try:
            raw_client.delete_collection(name)
        except Exception:
            pass

    # Clear stale sidecar so the next get_collection writes a fresh one.
    meta_path = os.path.join(palace_path, META_FILENAME)
    if os.path.exists(meta_path):
        os.remove(meta_path)

    # Recreate drawers collection via helper (fresh sidecar gets written).
    new_col = get_collection(
        palace_path,
        collection_name="mempalace_drawers",
        config=palace_config,
    )

    print(f"  Re-embedding {len(all_ids)} drawers...")
    filed = 0
    for i in range(0, len(all_ids), batch_size):
        new_col.add(
            ids=all_ids[i : i + batch_size],
            documents=all_docs[i : i + batch_size],
            metadatas=all_metas[i : i + batch_size],
        )
        filed += min(batch_size, len(all_ids) - i)
        print(f"    {filed}/{len(all_ids)}")

    # Re-embed compressed drawers if we had any.
    if comp_ids:
        new_comp = get_collection(
            palace_path,
            collection_name="mempalace_compressed",
            config=palace_config,
        )
        print(f"  Re-embedding {len(comp_ids)} compressed drawers...")
        filed = 0
        for i in range(0, len(comp_ids), batch_size):
            new_comp.add(
                ids=comp_ids[i : i + batch_size],
                documents=comp_docs[i : i + batch_size],
                metadatas=comp_metas[i : i + batch_size],
            )
            filed += min(batch_size, len(comp_ids) - i)
            print(f"    {filed}/{len(comp_ids)}")

    print(f"\n  Reembed complete.")
    print(f"  New dimension: {current_dim}")
    print(f"  New model:     {current_model}")
    print(f"  Backup saved:  {backup_path}")
    print(f"{'=' * 55}\n")


def main():
    parser = argparse.ArgumentParser(
        description="MemPalace — Give your AI a memory. No API key required.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--palace",
        default=None,
        help="Where the palace lives (default: from ~/.mempalace/config.json or ~/.mempalace/palace)",
    )

    sub = parser.add_subparsers(dest="command")

    # init
    p_init = sub.add_parser("init", help="Detect rooms from your folder structure")
    p_init.add_argument("dir", help="Project directory to set up")
    p_init.add_argument(
        "--yes", action="store_true", help="Auto-accept all detected entities (non-interactive)"
    )

    # mine
    p_mine = sub.add_parser("mine", help="Mine files into the palace")
    p_mine.add_argument("dir", help="Directory to mine")
    p_mine.add_argument(
        "--mode",
        choices=["projects", "convos"],
        default="projects",
        help="Ingest mode: 'projects' for code/docs (default), 'convos' for chat exports",
    )
    p_mine.add_argument("--wing", default=None, help="Wing name (default: directory name)")
    p_mine.add_argument(
        "--no-gitignore",
        action="store_true",
        help="Don't respect .gitignore files when scanning project files",
    )
    p_mine.add_argument(
        "--include-ignored",
        action="append",
        default=[],
        help="Always scan these project-relative paths even if ignored; repeat or pass comma-separated paths",
    )
    p_mine.add_argument(
        "--agent",
        default="mempalace",
        help="Your name — recorded on every drawer (default: mempalace)",
    )
    p_mine.add_argument("--limit", type=int, default=0, help="Max files to process (0 = all)")
    p_mine.add_argument(
        "--dry-run", action="store_true", help="Show what would be filed without filing"
    )
    p_mine.add_argument(
        "--extract",
        choices=["exchange", "general"],
        default="exchange",
        help="Extraction strategy for convos mode: 'exchange' (default) or 'general' (5 memory types)",
    )

    # search
    p_search = sub.add_parser("search", help="Find anything, exact words")
    p_search.add_argument("query", help="What to search for")
    p_search.add_argument("--wing", default=None, help="Limit to one project")
    p_search.add_argument("--room", default=None, help="Limit to one room")
    p_search.add_argument("--results", type=int, default=5, help="Number of results")
    rerank_group = p_search.add_mutually_exclusive_group()
    rerank_group.add_argument(
        "--rerank",
        action="store_true",
        default=None,
        help="Rerank results using the configured cross-encoder model",
    )
    rerank_group.add_argument(
        "--no-rerank",
        dest="rerank",
        action="store_false",
        help="Skip reranking even if configured",
    )

    # compress
    p_compress = sub.add_parser(
        "compress",
        help="Compress drawers using AAAK or Wenjian format",
    )
    p_compress.add_argument("--wing", default=None, help="Wing to compress (default: all wings)")
    p_compress.add_argument(
        "--dry-run", action="store_true", help="Preview compression without storing"
    )
    p_compress.add_argument(
        "--config", default=None, help="Entity config JSON (e.g. entities.json) — used by AAAK format"
    )
    p_compress.add_argument(
        "--format",
        choices=["aaak", "wenjian"],
        default=None,
        help="Compression format (default: from config, or 'aaak' for backward compat)",
    )
    p_compress.add_argument(
        "--rule-only",
        action="store_true",
        help="Use rule-based compression only, skip LLM even if configured",
    )

    # wake-up
    p_wakeup = sub.add_parser("wake-up", help="Show L0 + L1 wake-up context (~600-900 tokens)")
    p_wakeup.add_argument("--wing", default=None, help="Wake-up for a specific project/wing")

    # split
    p_split = sub.add_parser(
        "split",
        help="Split concatenated transcript mega-files into per-session files (run before mine)",
    )
    p_split.add_argument("dir", help="Directory containing transcript files")
    p_split.add_argument(
        "--output-dir",
        default=None,
        help="Write split files here (default: same directory as source files)",
    )
    p_split.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be split without writing files",
    )
    p_split.add_argument(
        "--min-sessions",
        type=int,
        default=2,
        help="Only split files containing at least N sessions (default: 2)",
    )

    # hook
    p_hook = sub.add_parser(
        "hook",
        help="Run hook logic (reads JSON from stdin, outputs JSON to stdout)",
    )
    hook_sub = p_hook.add_subparsers(dest="hook_action")
    p_hook_run = hook_sub.add_parser("run", help="Execute a hook")
    p_hook_run.add_argument(
        "--hook",
        required=True,
        choices=["session-start", "stop", "precompact"],
        help="Hook name to run",
    )
    p_hook_run.add_argument(
        "--harness",
        required=True,
        choices=["claude-code", "codex"],
        help="Harness type (determines stdin JSON format)",
    )

    # instructions
    p_instructions = sub.add_parser(
        "instructions",
        help="Output skill instructions to stdout",
    )
    instructions_sub = p_instructions.add_subparsers(dest="instructions_name")
    for instr_name in ["init", "search", "mine", "help", "status"]:
        instructions_sub.add_parser(instr_name, help=f"Output {instr_name} instructions")

    # repair
    sub.add_parser(
        "repair",
        help="Rebuild palace vector index from stored data (fixes segfaults after corruption)",
    )

    # reembed
    p_reembed = sub.add_parser(
        "reembed",
        help="Re-embed all drawers with the currently configured embedding provider",
    )
    p_reembed.add_argument(
        "--force",
        action="store_true",
        help="Skip the confirmation prompt and reembed unconditionally",
    )

    # mcp
    sub.add_parser(
        "mcp",
        help="Show MCP setup command for connecting MemPalace to your AI client",
    )

    # status
    sub.add_parser("status", help="Show what's been filed")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Handle two-level subcommands
    if args.command == "hook":
        if not getattr(args, "hook_action", None):
            p_hook.print_help()
            return
        cmd_hook(args)
        return

    if args.command == "instructions":
        name = getattr(args, "instructions_name", None)
        if not name:
            p_instructions.print_help()
            return
        args.name = name
        cmd_instructions(args)
        return

    dispatch = {
        "init": cmd_init,
        "mine": cmd_mine,
        "split": cmd_split,
        "search": cmd_search,
        "mcp": cmd_mcp,
        "compress": cmd_compress,
        "wake-up": cmd_wakeup,
        "repair": cmd_repair,
        "reembed": cmd_reembed,
        "status": cmd_status,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
