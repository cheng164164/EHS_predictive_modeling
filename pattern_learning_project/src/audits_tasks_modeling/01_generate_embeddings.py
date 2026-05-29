#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

import config as cfg

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from utils import ensure_dir, load_table, normalize_embeddings, save_csv, save_json


def embeddings_tfidf_svd(texts, n_components: int, max_features: int, min_df: int, ngram_max: int):
    n_components = max(2, int(n_components))
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=0.95,
        ngram_range=(1, ngram_max),
        stop_words="english",
        strip_accents="unicode",
        sublinear_tf=True,
    )
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    normalizer = Normalizer(copy=False)
    pipe = Pipeline([
        ("tfidf", vectorizer),
        ("svd", svd),
        ("normalizer", normalizer),
    ])
    x = pipe.fit_transform(texts)
    return np.asarray(x, dtype=np.float32), pipe


def embeddings_sentence_transformer(texts, model_name: str, batch_size: int):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    x = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(x, dtype=np.float32), model


def embeddings_azure_openai(texts, model_name: str, batch_size: int):
    from openai import AzureOpenAI
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
    if not endpoint or not api_key:
        raise EnvironmentError("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY to use azure_openai embeddings.")
    client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
    vectors = []
    texts = list(texts)
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = client.embeddings.create(model=model_name, input=batch)
        vectors.extend([d.embedding for d in response.data])
        print(f"embedded {min(start + batch_size, len(texts))}/{len(texts)}")
    return normalize_embeddings(np.asarray(vectors, dtype=np.float32)), {"provider": "azure_openai", "model": model_name}


def main():
    parser = argparse.ArgumentParser(description="Generate text embeddings for safety_text_event records.")
    parser.add_argument("--input", default=cfg.SAFETY_TEXT_EVENT_PATH)
    parser.add_argument("--output-dir", default=cfg.STEP_01_DIR)
    parser.add_argument("--provider", default=cfg.EMBEDDING_PROVIDER, choices=["tfidf_svd", "sentence_transformer", "azure_openai"])
    parser.add_argument("--model-name", default=cfg.EMBEDDING_MODEL_NAME, help="Azure deployment name or sentence-transformer model name.")
    parser.add_argument("--n-components", type=int, default=cfg.TFIDF_SVD_COMPONENTS)
    parser.add_argument("--max-features", type=int, default=cfg.TFIDF_MAX_FEATURES)
    parser.add_argument("--min-df", type=int, default=cfg.TFIDF_MIN_DF)
    parser.add_argument("--ngram-max", type=int, default=cfg.TFIDF_NGRAM_MAX)
    parser.add_argument("--batch-size", type=int, default=cfg.EMBEDDING_BATCH_SIZE)
    parser.add_argument("--text-column", default=cfg.TEXT_COLUMN)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    model_dir = ensure_dir(output_dir / "models")
    events = load_table(args.input)
    texts = events[args.text_column].fillna("").astype(str).tolist()

    if args.provider == "tfidf_svd":
        x, model = embeddings_tfidf_svd(texts, args.n_components, args.max_features, args.min_df, args.ngram_max)
        dump(model, model_dir / "embedding_model_tfidf_svd.joblib")
        model_info = {"provider": args.provider, "n_components": int(x.shape[1]), "max_features": args.max_features}
    elif args.provider == "sentence_transformer":
        model_name = args.model_name
        x, model = embeddings_sentence_transformer(texts, model_name, args.batch_size)
        model_info = {"provider": args.provider, "model_name": model_name, "n_components": int(x.shape[1])}
    else:
        x, model_info = embeddings_azure_openai(texts, args.model_name, args.batch_size)
        model_info["n_components"] = int(x.shape[1])

    x = normalize_embeddings(x)
    embeddings_path = output_dir / "text_embeddings.npy"
    id_map_path = output_dir / "text_embedding_event_ids.csv.gz"
    summary_path = output_dir / "01_embedding_summary.json"
    artifact_path = output_dir / "embedding_artifacts.json"

    np.save(embeddings_path, x.astype(np.float32))
    id_map = events[["event_id", "source_type", "source_id", "event_date"]].copy()
    save_csv(id_map, id_map_path)

    # These paths are used directly by Step 02 so it can reuse Step 01 output
    # instead of recomputing event embeddings. For tfidf_svd, the fitted
    # pipeline is also used to transform the small tag-definition library into
    # the exact same vector space as the event embeddings.
    artifacts = {
        **model_info,
        "embedding_shape": [int(x.shape[0]), int(x.shape[1])],
        "text_column": args.text_column,
        "embeddings_path": str(embeddings_path),
        "event_id_map_path": str(id_map_path),
        "model_dir": str(model_dir),
        "tfidf_svd_pipeline_path": str(model_dir / "embedding_model_tfidf_svd.joblib") if args.provider == "tfidf_svd" else None,
    }
    save_json(artifacts, summary_path)
    save_json(artifacts, artifact_path)
    print(artifacts)


if __name__ == "__main__":
    main()
