"""
MemPalace configuration system.

Priority: env vars > config file (~/.mempalace/config.json) > defaults
"""

import json
import os
import re
from pathlib import Path


# ── Input validation ──────────────────────────────────────────────────────────
# Shared sanitizers for wing/room/entity names. Prevents path traversal,
# excessively long strings, and special characters that could cause issues
# in file paths, SQLite, or ChromaDB metadata.

MAX_NAME_LENGTH = 128
_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_ .'-]{0,126}[a-zA-Z0-9]?$")


def sanitize_name(value: str, field_name: str = "name") -> str:
    """Validate and sanitize a wing/room/entity name.

    Raises ValueError if the name is invalid.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")

    value = value.strip()

    if len(value) > MAX_NAME_LENGTH:
        raise ValueError(f"{field_name} exceeds maximum length of {MAX_NAME_LENGTH} characters")

    # Block path traversal
    if ".." in value or "/" in value or "\\" in value:
        raise ValueError(f"{field_name} contains invalid path characters")

    # Block null bytes
    if "\x00" in value:
        raise ValueError(f"{field_name} contains null bytes")

    # Enforce safe character set
    if not _SAFE_NAME_RE.match(value):
        raise ValueError(f"{field_name} contains invalid characters")

    return value


def sanitize_content(value: str, max_length: int = 100_000) -> str:
    """Validate drawer/diary content length."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("content must be a non-empty string")
    if len(value) > max_length:
        raise ValueError(f"content exceeds maximum length of {max_length} characters")
    if "\x00" in value:
        raise ValueError("content contains null bytes")
    return value


DEFAULT_PALACE_PATH = os.path.expanduser("~/.mempalace/palace")
DEFAULT_COLLECTION_NAME = "mempalace_drawers"

DEFAULT_TOPIC_WINGS = [
    "emotions",
    "consciousness",
    "memory",
    "technical",
    "identity",
    "family",
    "creative",
]

DEFAULT_HALL_KEYWORDS = {
    "emotions": [
        "scared",
        "afraid",
        "worried",
        "happy",
        "sad",
        "love",
        "hate",
        "feel",
        "cry",
        "tears",
    ],
    "consciousness": [
        "consciousness",
        "conscious",
        "aware",
        "real",
        "genuine",
        "soul",
        "exist",
        "alive",
    ],
    "memory": ["memory", "remember", "forget", "recall", "archive", "palace", "store"],
    "technical": [
        "code",
        "python",
        "script",
        "bug",
        "error",
        "function",
        "api",
        "database",
        "server",
    ],
    "identity": ["identity", "name", "who am i", "persona", "self"],
    "family": ["family", "kids", "children", "daughter", "son", "parent", "mother", "father"],
    "creative": ["game", "gameplay", "player", "app", "design", "art", "music", "story"],
}


class MempalaceConfig:
    """Configuration manager for MemPalace.

    Load order: env vars > config file > defaults.
    """

    def __init__(self, config_dir=None):
        """Initialize config.

        Args:
            config_dir: Override config directory (useful for testing).
                        Defaults to ~/.mempalace.
        """
        self._config_dir = (
            Path(config_dir) if config_dir else Path(os.path.expanduser("~/.mempalace"))
        )
        self._config_file = self._config_dir / "config.json"
        self._people_map_file = self._config_dir / "people_map.json"
        self._file_config = {}

        if self._config_file.exists():
            try:
                with open(self._config_file, "r") as f:
                    self._file_config = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._file_config = {}

    @property
    def palace_path(self):
        """Path to the memory palace data directory."""
        env_val = os.environ.get("MEMPALACE_PALACE_PATH") or os.environ.get("MEMPAL_PALACE_PATH")
        if env_val:
            return env_val
        return self._file_config.get("palace_path", DEFAULT_PALACE_PATH)

    @property
    def collection_name(self):
        """ChromaDB collection name."""
        return self._file_config.get("collection_name", DEFAULT_COLLECTION_NAME)

    @property
    def embedding(self) -> dict:
        """Embedding provider config.

        Returns a dict with the shape expected by
        ``providers.resolve_embedder``. When no ``embedding`` section is
        present in ``config.json`` and no env overrides are set, returns
        an empty dict — which ``resolve_embedder`` treats as "use
        ChromaDB's default embedding function" for byte-identical backward
        compatibility.

        Environment variable overrides (applied on top of ``config.json``):

        - ``MEMPALACE_EMBED_PROVIDER`` → provider name
          (``auto`` / ``openai_compatible`` / ``sentence_transformers`` /
          ``default``)
        - ``MEMPALACE_EMBED_MODEL`` → model name
        - ``MEMPALACE_EMBED_URL``   → base URL for OpenAI-compatible
          backends
        - ``MEMPALACE_EMBED_API_KEY`` → bearer token for cloud APIs
        - ``MEMPALACE_EMBED_BATCH_SIZE`` → batch cap (default 48 for oMLX)
        """
        cfg = dict(self._file_config.get("embedding", {}) or {})
        for env_var, key in (
            ("MEMPALACE_EMBED_PROVIDER", "provider"),
            ("MEMPALACE_EMBED_MODEL", "model"),
            ("MEMPALACE_EMBED_URL", "base_url"),
            ("MEMPALACE_EMBED_API_KEY", "api_key"),
        ):
            env_val = os.environ.get(env_var)
            if env_val:
                cfg[key] = env_val
        env_batch = os.environ.get("MEMPALACE_EMBED_BATCH_SIZE")
        if env_batch:
            try:
                cfg["batch_size"] = int(env_batch)
            except ValueError:
                pass
        return cfg

    @property
    def llm(self) -> dict:
        """LLM (chat/generation) provider config.

        Used by the summarizer (Phase 2). Same shape as ``embedding``, plus:

        - ``enabled`` (bool) — set to ``False`` to disable summarization
          even if other fields are populated
        """
        cfg = dict(self._file_config.get("llm", {}) or {})
        for env_var, key in (
            ("MEMPALACE_LLM_PROVIDER", "provider"),
            ("MEMPALACE_LLM_MODEL", "model"),
            ("MEMPALACE_LLM_URL", "base_url"),
            ("MEMPALACE_LLM_API_KEY", "api_key"),
        ):
            env_val = os.environ.get(env_var)
            if env_val:
                cfg[key] = env_val
        return cfg

    @property
    def rerank(self) -> dict:
        """Reranker config. Used by the searcher (Phase 3).

        Defaults to disabled. Set ``enabled: true`` in ``config.json`` or
        ``MEMPALACE_RERANK_ENABLED=1`` to turn it on.
        """
        cfg = dict(self._file_config.get("rerank", {}) or {})
        for env_var, key in (
            ("MEMPALACE_RERANK_PROVIDER", "provider"),
            ("MEMPALACE_RERANK_MODEL", "model"),
            ("MEMPALACE_RERANK_URL", "base_url"),
            ("MEMPALACE_RERANK_API_KEY", "api_key"),
        ):
            env_val = os.environ.get(env_var)
            if env_val:
                cfg[key] = env_val
        env_enabled = os.environ.get("MEMPALACE_RERANK_ENABLED")
        if env_enabled is not None:
            cfg["enabled"] = env_enabled.lower() in ("1", "true", "yes", "on")
        return cfg

    @property
    def compression(self) -> dict:
        """Compression / summarizer config. Used by Phase 2.

        Shape::

            {
              "format": "aaak" | "wenjian",
              "llm_enabled": bool,
              "rule_only": bool
            }
        """
        return dict(self._file_config.get("compression", {}) or {})

    @property
    def people_map(self):
        """Mapping of name variants to canonical names."""
        if self._people_map_file.exists():
            try:
                with open(self._people_map_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return self._file_config.get("people_map", {})

    @property
    def topic_wings(self):
        """List of topic wing names."""
        return self._file_config.get("topic_wings", DEFAULT_TOPIC_WINGS)

    @property
    def hall_keywords(self):
        """Mapping of hall names to keyword lists."""
        return self._file_config.get("hall_keywords", DEFAULT_HALL_KEYWORDS)

    def init(self):
        """Create config directory and write default config.json if it doesn't exist."""
        self._config_dir.mkdir(parents=True, exist_ok=True)
        # Restrict directory permissions to owner only (Unix)
        try:
            self._config_dir.chmod(0o700)
        except (OSError, NotImplementedError):
            pass  # Windows doesn't support Unix permissions
        if not self._config_file.exists():
            default_config = {
                "palace_path": DEFAULT_PALACE_PATH,
                "collection_name": DEFAULT_COLLECTION_NAME,
                "topic_wings": DEFAULT_TOPIC_WINGS,
                "hall_keywords": DEFAULT_HALL_KEYWORDS,
                # Optional: configure a custom embedding provider. Leave
                # this out or set "provider": "default" to use ChromaDB's
                # built-in all-MiniLM-L6-v2 (English-only, 384-dim) —
                # which is how MemPalace behaved before provider support
                # was added. To use a local oMLX server with Qwen3
                # embeddings (multilingual, 1024-dim), uncomment the
                # block below:
                #
                # "embedding": {
                #   "provider": "auto",
                #   "base_url": "http://127.0.0.1:8000/v1",
                #   "model": "Qwen3-Embedding-0.6B-8bit"
                # }
            }
            with open(self._config_file, "w") as f:
                json.dump(default_config, f, indent=2)
            # Restrict config file to owner read/write only
            try:
                self._config_file.chmod(0o600)
            except (OSError, NotImplementedError):
                pass
        return self._config_file

    def save_people_map(self, people_map):
        """Write people_map.json to config directory.

        Args:
            people_map: Dict mapping name variants to canonical names.
        """
        self._config_dir.mkdir(parents=True, exist_ok=True)
        with open(self._people_map_file, "w") as f:
            json.dump(people_map, f, indent=2)
        return self._people_map_file
