"""
Dataset loading and processing utilities for ACE-Code.

This module provides utilities for loading code datasets
from various sources (ManyTypes4Py, The Stack, MBPP, etc.)
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import torch
from loguru import logger
from torch.utils.data import Dataset, DataLoader

try:
    from datasets import load_dataset, Dataset as HFDataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False


@dataclass
class CodeSample:
    """A single code sample for analysis."""

    code: str
    language: str = "python"
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


class CodeDataset(Dataset):
    """
    PyTorch Dataset for code samples.

    This provides a standard interface for iterating over code samples
    for batch processing during circuit analysis.
    """

    def __init__(
        self,
        samples: List[CodeSample],
        tokenizer: Any = None,
        max_length: int = 512,
    ):
        """
        Initialize the dataset.

        Args:
            samples: List of CodeSample objects
            tokenizer: Tokenizer for encoding (optional)
            max_length: Maximum sequence length
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        item = {
            "code": sample.code,
            "language": sample.language,
            "source": sample.source,
            "idx": idx,
        }

        # Add tokenization if tokenizer provided
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                sample.code,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
            )
            item["input_ids"] = tokens["input_ids"].squeeze(0)
            item["attention_mask"] = tokens["attention_mask"].squeeze(0)

        return item

    @classmethod
    def from_list(
        cls,
        code_strings: List[str],
        language: str = "python",
        source: str = "custom",
        **kwargs,
    ) -> "CodeDataset":
        """Create dataset from a list of code strings."""
        samples = [
            CodeSample(code=code, language=language, source=source)
            for code in code_strings
        ]
        return cls(samples, **kwargs)


def load_code_dataset(
    name: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    language: str = "python",
    cache_dir: Optional[str] = None,
) -> CodeDataset:
    """
    Load a code dataset from HuggingFace or local files.

    Supported datasets:
    - manytypes4py: Python code with type annotations
    - the-stack: Multi-language code from GitHub
    - mbpp: Mostly Basic Python Problems
    - humaneval: HumanEval benchmark

    Args:
        name: Dataset name or path
        split: Dataset split (train, test, validation)
        max_samples: Maximum number of samples to load
        language: Language filter (for multi-language datasets)
        cache_dir: Directory to cache downloaded datasets

    Returns:
        CodeDataset instance
    """
    if not HF_DATASETS_AVAILABLE:
        raise ImportError("datasets library required: pip install datasets")

    logger.info(f"Loading dataset: {name} ({split})")

    samples = []

    try:
        if name.lower() == "manytypes4py":
            # ManyTypes4Py - Python with type annotations
            ds = load_dataset(
                "kevinjohnsonpro/manytypes4py",
                split=split,
                cache_dir=cache_dir,
            )
            for item in ds:
                if max_samples and len(samples) >= max_samples:
                    break
                samples.append(CodeSample(
                    code=item.get("content", item.get("code", "")),
                    language="python",
                    source="manytypes4py",
                    metadata={"types": item.get("types", [])},
                ))

        elif name.lower() == "the-stack" or name.lower() == "bigcode/the-stack":
            # The Stack - multi-language
            ds = load_dataset(
                "bigcode/the-stack",
                data_dir=f"data/{language}",
                split=split,
                cache_dir=cache_dir,
                streaming=True,  # Use streaming for large dataset
            )
            for item in ds:
                if max_samples and len(samples) >= max_samples:
                    break
                samples.append(CodeSample(
                    code=item["content"],
                    language=language,
                    source="the-stack",
                    metadata={"path": item.get("path", "")},
                ))

        elif name.lower() == "mbpp":
            # MBPP - Mostly Basic Python Problems
            ds = load_dataset(
                "mbpp",
                split=split,
                cache_dir=cache_dir,
            )
            for item in ds:
                if max_samples and len(samples) >= max_samples:
                    break
                samples.append(CodeSample(
                    code=item["code"],
                    language="python",
                    source="mbpp",
                    metadata={
                        "task_id": item["task_id"],
                        "text": item["text"],
                        "test_list": item.get("test_list", []),
                    },
                ))

        elif name.lower() == "humaneval":
            # HumanEval benchmark
            ds = load_dataset(
                "openai_humaneval",
                split=split,
                cache_dir=cache_dir,
            )
            for item in ds:
                if max_samples and len(samples) >= max_samples:
                    break
                samples.append(CodeSample(
                    code=item["prompt"] + item["canonical_solution"],
                    language="python",
                    source="humaneval",
                    metadata={
                        "task_id": item["task_id"],
                        "prompt": item["prompt"],
                        "entry_point": item["entry_point"],
                    },
                ))

        else:
            # Try to load as a generic HuggingFace dataset
            ds = load_dataset(name, split=split, cache_dir=cache_dir)

            # Try to find the code field
            code_fields = ["code", "content", "text", "source_code", "func_code"]

            for item in ds:
                if max_samples and len(samples) >= max_samples:
                    break

                code = None
                for field in code_fields:
                    if field in item:
                        code = item[field]
                        break

                if code:
                    samples.append(CodeSample(
                        code=code,
                        language=language,
                        source=name,
                    ))

    except Exception as e:
        logger.error(f"Error loading dataset {name}: {e}")
        raise

    logger.info(f"Loaded {len(samples)} samples from {name}")
    return CodeDataset(samples)


def load_from_directory(
    directory: Union[str, Path],
    extensions: List[str] = [".py"],
    recursive: bool = True,
    max_samples: Optional[int] = None,
) -> CodeDataset:
    """
    Load code samples from a local directory.

    Args:
        directory: Path to directory containing code files
        extensions: File extensions to include
        recursive: Whether to search recursively
        max_samples: Maximum number of files to load

    Returns:
        CodeDataset instance
    """
    directory = Path(directory)
    samples = []

    pattern = "**/*" if recursive else "*"

    for ext in extensions:
        for filepath in directory.glob(pattern + ext):
            if max_samples and len(samples) >= max_samples:
                break

            try:
                code = filepath.read_text(encoding="utf-8")
                lang = {
                    ".py": "python",
                    ".js": "javascript",
                    ".ts": "typescript",
                    ".java": "java",
                    ".cpp": "cpp",
                    ".c": "c",
                    ".go": "go",
                    ".rs": "rust",
                }.get(ext, "unknown")

                samples.append(CodeSample(
                    code=code,
                    language=lang,
                    source=str(filepath),
                    metadata={"filepath": str(filepath)},
                ))
            except Exception as e:
                logger.warning(f"Error reading {filepath}: {e}")

    logger.info(f"Loaded {len(samples)} files from {directory}")
    return CodeDataset(samples)


def create_dataloader(
    dataset: CodeDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader for a CodeDataset.

    Args:
        dataset: CodeDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes

    Returns:
        PyTorch DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_code_samples,
    )


def _collate_code_samples(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for batching code samples."""
    collated = {
        "code": [item["code"] for item in batch],
        "language": [item["language"] for item in batch],
        "source": [item["source"] for item in batch],
        "idx": [item["idx"] for item in batch],
    }

    # Stack tensors if present
    if "input_ids" in batch[0]:
        collated["input_ids"] = torch.stack([item["input_ids"] for item in batch])
        collated["attention_mask"] = torch.stack([item["attention_mask"] for item in batch])

    return collated
