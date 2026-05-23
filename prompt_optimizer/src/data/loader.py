"""PDF and gold JSON data loading with deterministic train/val/test splits."""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Document:
    """A single document with its PDF path and gold annotation."""

    doc_id: str
    pdf_path: Path
    gold_path: Path
    gold_data: dict = field(repr=False)


@dataclass
class DataSplit:
    """Deterministic train/validation/test split of documents."""

    train: list[Document]
    val: list[Document]
    test: list[Document]
    schema_name: str

    @property
    def all_documents(self) -> list[Document]:
        """Return all documents across all three splits."""
        return self.train + self.val + self.test

    def summary(self) -> str:
        """Return a human-readable summary of the split."""
        return (
            f"Schema: {self.schema_name} | "
            f"Train: {len(self.train)} | "
            f"Val: {len(self.val)} | "
            f"Test: {len(self.test)} | "
            f"Total: {len(self.all_documents)}"
        )


# Maps schema identifier to (domain, schema_dir_name) tuple.
SCHEMA_PATHS: dict[str, tuple[str, str]] = {
    "academic/research": ("academic", "research"),
    "finance/10kq": ("finance", "10kq"),
    "finance/credit_agreement": ("finance", "credit_agreement"),
    "hiring/resume": ("hiring", "resume"),
    "sport/swimming": ("sport", "swimming"),
}

# Maps schema identifier to the schema JSON filename.
SCHEMA_FILE_NAMES: dict[str, str] = {
    "academic/research": "research-schema.json",
    "finance/10kq": "10kq-schema.json",
    "finance/credit_agreement": "credit_agreement-schema.json",
    "hiring/resume": "resume-schema.json",
    "sport/swimming": "swimming-schema.json",
}


def get_schema_dir(dataset_path: str, schema_name: str) -> Path:
    """Resolve the filesystem directory for a given schema identifier.

    Args:
        dataset_path: Root path to the ExtractBench dataset directory.
        schema_name: Schema identifier such as 'academic/research'.

    Returns:
        Absolute path to the schema directory.

    Raises:
        ValueError: If schema_name is not recognized.
        FileNotFoundError: If the directory does not exist on disk.
    """
    if schema_name not in SCHEMA_PATHS:
        valid = ", ".join(sorted(SCHEMA_PATHS.keys()))
        raise ValueError(
            f"Unknown schema '{schema_name}'. Valid schemas: {valid}"
        )
    domain, schema_dir_name = SCHEMA_PATHS[schema_name]
    schema_dir = Path(dataset_path) / domain / schema_dir_name
    if not schema_dir.exists():
        raise FileNotFoundError(f"Schema directory not found: {schema_dir}")
    return schema_dir


def get_schema_file_path(dataset_path: str, schema_name: str) -> Path:
    """Get the path to the schema JSON file for a given schema.

    Args:
        dataset_path: Root path to the ExtractBench dataset directory.
        schema_name: Schema identifier such as 'academic/research'.

    Returns:
        Path to the schema JSON file.

    Raises:
        FileNotFoundError: If the schema file does not exist.
    """
    schema_dir = get_schema_dir(dataset_path, schema_name)
    filename = SCHEMA_FILE_NAMES.get(schema_name)

    if filename:
        schema_file = schema_dir / filename
    else:
        candidates = list(schema_dir.glob("*-schema.json"))
        if not candidates:
            raise FileNotFoundError(
                f"No *-schema.json file found in {schema_dir}"
            )
        schema_file = candidates[0]

    if not schema_file.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_file}")
    return schema_file


def discover_documents(dataset_path: str, schema_name: str) -> list[Document]:
    """Discover all PDF + gold-JSON document pairs for a schema.

    Scans the ``pdf+gold/`` directory (non-recursively) so that
    subdirectories like ``_extra/`` are automatically excluded.

    Args:
        dataset_path: Root path to the ExtractBench dataset directory.
        schema_name: Schema identifier such as 'academic/research'.

    Returns:
        Sorted list of Document objects (sorted by doc_id).

    Raises:
        FileNotFoundError: If the pdf+gold directory is missing.
        ValueError: If no document pairs are found.
    """
    schema_dir = get_schema_dir(dataset_path, schema_name)
    pdf_gold_dir = schema_dir / "pdf+gold"

    if not pdf_gold_dir.exists():
        raise FileNotFoundError(
            f"pdf+gold directory not found: {pdf_gold_dir}"
        )

    documents: list[Document] = []
    for gold_path in sorted(pdf_gold_dir.glob("*.gold.json")):
        doc_id = gold_path.name.removesuffix(".gold.json")
        pdf_path = gold_path.parent / f"{doc_id}.pdf"

        if not pdf_path.exists():
            print(f"[WARN] PDF missing for '{doc_id}', skipping.")
            continue

        gold_data = _load_json(gold_path)
        documents.append(
            Document(
                doc_id=doc_id,
                pdf_path=pdf_path,
                gold_path=gold_path,
                gold_data=gold_data,
            )
        )

    if not documents:
        raise ValueError(f"No document pairs found in {pdf_gold_dir}")

    return documents


def split_documents(
    documents: list[Document],
    seed: int = 42,
    ratios: list[float] | None = None,
) -> tuple[list[Document], list[Document], list[Document]]:
    """Split documents into train / val / test deterministically.

    Algorithm:
      1. Sort documents alphabetically by doc_id.
      2. Shuffle with a seeded RNG (``random.Random(seed)``).
      3. Allocate val and test sizes first (``max(1, round(n * ratio))``),
         then assign the remainder to train.

    Args:
        documents: List of documents to split.
        seed: Random seed for reproducibility.
        ratios: ``[train, val, test]`` fractions; must sum to ~1.0.

    Returns:
        ``(train, val, test)`` tuple of document lists.
    """
    if ratios is None:
        ratios = [0.7, 0.15, 0.15]

    if len(ratios) != 3:
        raise ValueError("ratios must contain exactly 3 values")

    n = len(documents)
    if n < 3:
        raise ValueError(f"Need at least 3 documents, got {n}")

    sorted_docs = sorted(documents, key=lambda d: d.doc_id)
    shuffled = list(sorted_docs)
    random.Random(seed).shuffle(shuffled)

    val_size = max(1, round(n * ratios[1]))
    test_size = max(1, round(n * ratios[2]))
    train_size = n - val_size - test_size

    if train_size < 1:
        train_size = 1
        remaining = n - 1
        val_size = remaining // 2
        test_size = remaining - val_size

    train = shuffled[:train_size]
    val = shuffled[train_size : train_size + val_size]
    test = shuffled[train_size + val_size :]

    return train, val, test


class DatasetLoader:
    """High-level loader for ExtractBench datasets with deterministic splits."""

    def __init__(self, dataset_path: str, schema_name: str) -> None:
        """Initialise the loader.

        Args:
            dataset_path: Root path to the ExtractBench ``dataset/`` directory.
            schema_name: Schema identifier (e.g. ``'academic/research'``).
        """
        self.dataset_path = dataset_path
        self.schema_name = schema_name
        self._schema_dir = get_schema_dir(dataset_path, schema_name)

    def load(
        self,
        split_seed: int = 42,
        split_ratios: list[float] | None = None,
        max_docs_per_split: int | None = None,
    ) -> DataSplit:
        """Load documents and produce a deterministic train/val/test split.

        Args:
            split_seed: Random seed for the split.
            split_ratios: ``[train, val, test]`` fractions.
            max_docs_per_split: If set, truncate each split to at most
                this many documents (useful for ``--dry-run``).

        Returns:
            A ``DataSplit`` instance.
        """
        documents = discover_documents(self.dataset_path, self.schema_name)
        train, val, test = split_documents(documents, split_seed, split_ratios)

        if max_docs_per_split is not None:
            train = train[:max_docs_per_split]
            val = val[:max_docs_per_split]
            test = test[:max_docs_per_split]

        return DataSplit(
            train=train,
            val=val,
            test=test,
            schema_name=self.schema_name,
        )

    def get_schema_path(self) -> Path:
        """Return the filesystem path to the schema JSON file."""
        return get_schema_file_path(self.dataset_path, self.schema_name)


def _load_json(path: Path) -> dict:
    """Load a JSON file and return its contents as a dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
