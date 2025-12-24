"""Evaluate multilingual duplicate detection on a curated sample set."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
try:
    from sentence_transformers import SentenceTransformer, util as st_util
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore
    st_util = None  # type: ignore

# Allow imports from streamlit-app
ROOT = Path(__file__).parent
sys.path.append(str(ROOT / "streamlit-app"))
from language_utils import primary_language  # noqa: E402


def load_encoder():
    if SentenceTransformer is None:
        return None
    return SentenceTransformer("distiluse-base-multilingual-cased-v2")


def similarity_score(encoder, q1: str, q2: str) -> float:
    if encoder is not None and st_util is not None:
        embeddings = encoder.encode([q1, q2])
        sim = float(st_util.cos_sim(embeddings[0], embeddings[1]))
        return (sim + 1) / 2  # map to [0, 1]

    # Lightweight overlap-based similarity fallback
    toks1 = set(q1.lower().split())
    toks2 = set(q2.lower().split())
    if not toks1 or not toks2:
        return 0.0
    overlap = len(toks1 & toks2) / len(toks1 | toks2)
    return float(overlap)


def evaluate(df: pd.DataFrame, encoder) -> Tuple[float, List[float]]:
    scores: List[float] = []
    preds: List[int] = []
    for _, row in df.iterrows():
        score = similarity_score(encoder, row["q1"], row["q2"])
        scores.append(score)
        preds.append(1 if score >= 0.5 else 0)
    accuracy = float((np.array(preds) == df["is_duplicate"].to_numpy()).mean())
    return accuracy, scores


def summarize_languages(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        q1_lang = primary_language(row["q1"])
        q2_lang = primary_language(row["q2"])
        records.append({
            "q1_lang": q1_lang or "unknown",
            "q2_lang": q2_lang or "unknown",
            "pair_language": row.get("language", "unknown"),
        })
    return pd.DataFrame(records)


def main():
    data_path = ROOT / "data/multilingual/eval.csv"
    df = pd.read_csv(data_path)

    print(f"Loaded {len(df)} multilingual examples from {data_path}.")
    encoder = load_encoder()
    if encoder is None:
        print("sentence-transformers not available; using token-overlap fallback for evaluation.")

    accuracy, scores = evaluate(df, encoder)
    print(f"Similarity-threshold accuracy: {accuracy:.3f}")
    print(f"Mean similarity score: {np.mean(scores):.3f} | Std: {np.std(scores):.3f}")

    lang_summary = summarize_languages(df)
    lang_counts = lang_summary["pair_language"].value_counts().to_dict()
    print("Language counts:", lang_counts)

    detection_coverage = (lang_summary[["q1_lang", "q2_lang"]] != "unknown").mean().mean()
    print(f"Language detection coverage across questions: {detection_coverage*100:.1f}%")


if __name__ == "__main__":
    main()
