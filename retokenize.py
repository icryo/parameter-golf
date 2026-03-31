#!/usr/bin/env python3
"""Retokenize FineWeb data with a custom TokenMonster vocabulary.

Usage:
  python3 retokenize.py --vocab candidate.vocab --output-dir data/datasets/fineweb10B_custom

This:
1. Downloads raw FineWeb docs (docs_selected.jsonl) from HuggingFace
2. Tokenizes each document with the specified TokenMonster vocab
3. Exports in the competition's binary shard format (magic 20240520)
4. Generates per-token metadata (byte lengths, leading spaces, boundaries)
5. Exports metadata as .meta.npz for BPP evaluation
"""
from __future__ import annotations
import argparse
import json
import os
import struct
import sys
from pathlib import Path
import numpy as np
import tokenmonster
from huggingface_hub import hf_hub_download

REPO_ID = os.environ.get("MATCHED_FINEWEB_REPO_ID", "willdepueoai/parameter-golf")
DOCS_FILENAME = "docs_selected.jsonl"
DATAFILE_MAGIC = 20240520
DATAFILE_VERSION = 1
SHARD_SIZE = 100_000_000  # tokens per shard
NUM_VAL_DOCS = 50_000


def download_docs(cache_dir: Path) -> Path:
    """Download the raw FineWeb docs JSONL from HuggingFace."""
    docs_path = cache_dir / DOCS_FILENAME
    if docs_path.exists():
        print(f"Docs already downloaded: {docs_path}")
        return docs_path
    print(f"Downloading {DOCS_FILENAME} from {REPO_ID}...")
    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        filename=DOCS_FILENAME,
        repo_type="model",
        local_dir=str(cache_dir),
    )
    return Path(downloaded)


def build_token_metadata(vocab: tokenmonster.Vocab) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-token byte length, leading space, and boundary LUTs."""
    vocab_size = len(vocab)
    base_bytes = np.zeros(vocab_size, dtype=np.int16)
    has_leading_space = np.zeros(vocab_size, dtype=np.bool_)
    is_boundary = np.ones(vocab_size, dtype=np.bool_)  # default: boundary

    for token_id in range(vocab_size):
        try:
            token_str = vocab.decode([token_id])
            if not token_str:
                continue
            is_boundary[token_id] = False
            token_bytes = token_str.encode('utf-8')
            base_bytes[token_id] = len(token_bytes)
            if token_str.startswith(' '):
                has_leading_space[token_id] = True
                # Adjust: the leading space byte is counted separately
                # (matches SentencePiece's ▁ handling)
        except Exception:
            continue

    return base_bytes, has_leading_space, is_boundary


def save_metadata(path: Path, vocab: tokenmonster.Vocab, vocab_name: str):
    """Save tokenizer metadata as .meta.npz for runtime byte accounting."""
    base_bytes, has_leading_space, is_boundary = build_token_metadata(vocab)
    np.savez(
        path,
        format_version=np.int32(1),
        vocab_size=np.int32(len(vocab)),
        tokenizer_kind=np.array("tokenmonster"),
        source_model_name=np.array(vocab_name),
        base_bytes=base_bytes,
        has_leading_space=has_leading_space,
        is_boundary_token=is_boundary,
    )
    print(f"Metadata saved: {path}")


def write_shard(path: Path, tokens: np.ndarray):
    """Write tokens in competition binary format."""
    header = np.zeros(256, dtype=np.int32)
    header[0] = DATAFILE_MAGIC
    header[1] = DATAFILE_VERSION
    header[2] = len(tokens)
    with open(path, 'wb') as f:
        f.write(header.tobytes())
        f.write(tokens.astype(np.uint16).tobytes())
    print(f"  Shard: {path.name} ({len(tokens):,} tokens, {path.stat().st_size / 1e6:.1f} MB)")


def retokenize(
    docs_path: Path,
    vocab: tokenmonster.Vocab,
    output_dir: Path,
    num_val_docs: int = NUM_VAL_DOCS,
    shard_size: int = SHARD_SIZE,
):
    """Retokenize all documents and write train/val shards."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Count total docs first
    print("Counting documents...")
    total_docs = sum(1 for _ in open(docs_path))
    train_docs = total_docs - num_val_docs
    print(f"Total docs: {total_docs:,} (train: {train_docs:,}, val: {num_val_docs:,})")

    # Process documents
    all_tokens = []
    current_shard = []
    shard_idx = 0
    doc_idx = 0
    phase = "train"
    total_tokens = 0

    with open(docs_path) as f:
        for line in f:
            doc = json.loads(line)
            text = doc.get("text", "")
            if not text:
                continue

            tokens = vocab.tokenize(text)
            token_list = tokens.tolist() if hasattr(tokens, 'tolist') else list(tokens)
            current_shard.extend(token_list)
            total_tokens += len(token_list)

            doc_idx += 1

            # Switch to val after train_docs
            if doc_idx == train_docs and phase == "train":
                # Flush remaining train tokens
                if current_shard:
                    shard_tokens = np.array(current_shard, dtype=np.uint16)
                    shard_path = output_dir / f"fineweb_train_{shard_idx:06d}.bin"
                    write_shard(shard_path, shard_tokens)
                    shard_idx += 1
                    current_shard = []
                print(f"\nTrain complete: {shard_idx} shards, {total_tokens:,} tokens")
                phase = "val"
                total_tokens = 0
                shard_idx = 0
                continue

            # Write shard when full (train phase only)
            if phase == "train" and len(current_shard) >= shard_size:
                shard_tokens = np.array(current_shard[:shard_size], dtype=np.uint16)
                shard_path = output_dir / f"fineweb_train_{shard_idx:06d}.bin"
                write_shard(shard_path, shard_tokens)
                current_shard = current_shard[shard_size:]
                shard_idx += 1

            if doc_idx % 100000 == 0:
                print(f"  Processed {doc_idx:,}/{total_docs:,} docs ({phase})...")

    # Write final val shard
    if current_shard:
        shard_tokens = np.array(current_shard, dtype=np.uint16)
        shard_path = output_dir / f"fineweb_val_{shard_idx:06d}.bin"
        write_shard(shard_path, shard_tokens)

    print(f"\nVal complete: {total_tokens:,} tokens")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Retokenize FineWeb with TokenMonster")
    parser.add_argument("--vocab", required=True, help="Path to .vocab file")
    parser.add_argument("--vocab-name", default="custom", help="Name for metadata")
    parser.add_argument("--output-dir", required=True, help="Output directory for shards")
    parser.add_argument("--cache-dir", default="./data/raw", help="Cache dir for raw docs")
    parser.add_argument("--val-docs", type=int, default=NUM_VAL_DOCS)
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer: {args.vocab}")
    vocab = tokenmonster.load(args.vocab)
    print(f"Vocab size: {len(vocab)}")

    # Download raw docs
    docs_path = download_docs(Path(args.cache_dir))

    # Build and save metadata
    meta_path = Path(args.output_dir) / "candidate.meta.npz"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    save_metadata(meta_path, vocab, args.vocab_name)

    # Copy vocab file
    import shutil
    vocab_dest = Path(args.output_dir) / "candidate.vocab"
    shutil.copy2(args.vocab, vocab_dest)
    print(f"Vocab copied: {vocab_dest}")

    # Retokenize
    retokenize(
        docs_path=docs_path,
        vocab=vocab,
        output_dir=Path(args.output_dir),
        num_val_docs=args.val_docs,
        shard_size=args.shard_size,
    )

    # Summary
    train_shards = list(Path(args.output_dir).glob("fineweb_train_*.bin"))
    val_shards = list(Path(args.output_dir).glob("fineweb_val_*.bin"))
    print(f"\n=== Complete ===")
    print(f"Train shards: {len(train_shards)}")
    print(f"Val shards: {len(val_shards)}")
    print(f"Metadata: {meta_path}")
    print(f"Vocab: {vocab_dest}")


if __name__ == "__main__":
    main()
