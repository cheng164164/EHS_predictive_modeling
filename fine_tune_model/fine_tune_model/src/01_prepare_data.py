import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import argparse
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD

from utils import cfg_path, compact_space, ensure_dir, load_config, load_yaml, safe_str, set_seed, write_jsonl

SYSTEM_PROMPT = (
    'You are a safety risk analysis assistant. Return concise, grounded, valid JSON. '
    'Do not invent facts not supported by the input. If details are missing, state the limitation.'
)

BASE_SCHEMA_KEYS = [
    'event_summary', 'risk_pattern', 'risk_pattern_description', 'additional_patterns', 'discovered_theme', 'hazard_tags', 'control_failure_tags',
    'potential_consequence', 'risk_level', 'recommended_actions', 'escalation_recommended',
    'recommended_review_group', 'evidence_phrases', 'limitations'
]

BASIC_SPANISH_STOPWORDS = {
    'a','al','algo','algunas','algunos','ante','antes','como','con','contra','cual','cuando','de','del',
    'desde','donde','durante','e','el','ella','ellas','ellos','en','entre','era','erais','eran','eras','eres',
    'es','esa','esas','ese','eso','esos','esta','estaba','estado','estais','estamos','estan','estar','este',
    'esto','estos','fue','fueron','ha','hace','hacen','hacer','hacia','han','hasta','hay','la','las','le','les',
    'lo','los','mas','me','mi','mis','mucho','muy','no','nos','o','otra','otras','otro','otros','para','pero',
    'por','porque','que','se','ser','si','sin','sobre','su','sus','tambien','tan','te','tener','todo','todos',
    'tu','un','una','uno','unos','y','ya'
}

BASIC_RUSSIAN_STOPWORDS = {
    'и','в','во','не','что','он','на','я','с','со','как','а','то','все','она','так','его','но','да','ты','к','у',
    'же','вы','за','бы','по','только','ее','мне','было','вот','от','меня','еще','нет','о','из','ему','теперь',
    'когда','даже','ну','вдруг','ли','если','уже','или','ни','быть','был','него','до','вас','нибудь','опять'
}


def strip_accents(text: str) -> str:
    return ''.join(ch for ch in unicodedata.normalize('NFKD', text) if not unicodedata.combining(ch))


def normalize_text(text: str, taxonomy: Dict[str, Any] | None = None) -> str:
    text = safe_str(text)
    if taxonomy:
        for pattern in taxonomy.get('text_cleaning', {}).get('field_noise_regexes', []):
            text = re.sub(pattern, ' ', text)
    text = strip_accents(text.lower())
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\b', ' ', text)
    text = re.sub(r'[^a-z0-9áéíóúüñа-яё/\- ]+', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{1,4}\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def simple_stem(token: str) -> str:
    # Dependency-free light stemming. This is intentionally conservative to avoid damaging safety terms.
    if len(token) <= 4:
        return token
    for suf in ['ciones', 'mente', 'acion', 'ments', 'tion', 'ing', 'ed', 'es', 's']:
        if token.endswith(suf) and len(token) - len(suf) >= 4:
            return token[:-len(suf)]
    return token


def build_stopwords(taxonomy: Dict[str, Any]) -> set[str]:
    custom = taxonomy.get('text_cleaning', {}).get('custom_stopwords', [])
    stops = set(ENGLISH_STOP_WORDS) | BASIC_SPANISH_STOPWORDS | BASIC_RUSSIAN_STOPWORDS
    stops |= {strip_accents(safe_str(x).lower()) for x in custom}
    stops |= {'na', 'n/a', 'none', 'null', 'unknown', 'yes', 'no', 'ok'}
    return {s for s in stops if s}


def tokenizer_factory(stopwords: set[str], use_stemming: bool = True):
    def tokenize(text: str) -> List[str]:
        text = normalize_text(text)
        toks = re.findall(r'[a-záéíóúüñа-яё][a-záéíóúüñа-яё\-]{2,}', text, flags=re.IGNORECASE)
        clean = []
        for tok in toks:
            tok = strip_accents(tok.lower()).strip('-')
            if not tok or tok in stopwords or len(tok) < 3:
                continue
            if re.fullmatch(r'(.)\1{2,}', tok):
                continue
            clean.append(simple_stem(tok) if use_stemming else tok)
        return clean
    return tokenize


@lru_cache(maxsize=10000)
def normalize_keyword(kw: str) -> str:
    return normalize_text(kw)

def keyword_score(text_norm: str, keywords: List[str], phrase_weight: float = 2.0) -> Tuple[float, List[str]]:
    score = 0.0
    hits = []
    for kw in keywords:
        kw_norm = normalize_keyword(safe_str(kw))
        if not kw_norm:
            continue
        if ' ' in kw_norm or '-' in kw_norm or '/' in kw_norm:
            if kw_norm in text_norm:
                score += phrase_weight
                hits.append(kw)
        else:
            if re.search(rf'\b{re.escape(kw_norm)}\b', text_norm):
                score += 1.0
                hits.append(kw)
    return score, hits


def infer_pattern(text: str, source_type: str, taxonomy: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[str, List[str], str, float]:
    text_norm = normalize_text(text, taxonomy)
    phrase_weight = float(cfg.get('labeling', {}).get('taxonomy', {}).get('phrase_weight', 2.0))
    min_score = float(cfg.get('labeling', {}).get('taxonomy', {}).get('min_keyword_score', 1))
    best = (0.0, 'general safety observation', [], 'injury or loss event depending on exposure and control effectiveness')
    for fam in taxonomy.get('hazard_families', []):
        score, hits = keyword_score(text_norm, fam.get('keywords', []), phrase_weight=phrase_weight)
        if score > best[0]:
            best = (score, fam.get('name', 'general safety observation'), hits, fam.get('consequence', 'injury or loss event depending on exposure and control effectiveness'))
    if best[0] < min_score:
        if source_type == 'task':
            return 'corrective action / administrative follow-up', [], 'injury or loss event depending on closure quality and related exposure', 0.0
        if source_type == 'audit':
            return 'audit observation / control verification', [], 'injury or loss event depending on exposure and control effectiveness', 0.0
        return 'general safety observation', [], 'injury or loss event depending on exposure and control effectiveness', 0.0
    return best[1], best[2][:8], best[3], best[0]


def infer_controls(text: str, taxonomy: Dict[str, Any]) -> List[str]:
    text_norm = normalize_text(text, taxonomy)
    scored = []
    for ctl in taxonomy.get('control_families', []):
        score, _ = keyword_score(text_norm, ctl.get('keywords', []), phrase_weight=2.0)
        if score > 0:
            scored.append((score, ctl.get('name', 'control gap')))
    scored.sort(reverse=True)
    return [x[1] for x in scored[:5]] or ['control effectiveness to be reviewed']


def get_family(taxonomy: Dict[str, Any], name: str) -> Dict[str, Any]:
    for fam in taxonomy.get('hazard_families', []):
        if fam.get('name') == name:
            return fam
    return {}


def as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    s = str(x).strip().lower()
    return s in {'1', 'true', 'yes', 'y', 't'}


def infer_risk_level(row: pd.Series, pattern: str, text: str, taxonomy: Dict[str, Any]) -> str:
    if as_bool(row.get('severe_actual')):
        return 'critical'
    fam = get_family(taxonomy, pattern)
    severity_hint = fam.get('severity_hint')
    t = normalize_text(text, taxonomy)
    high_exposure_words = ['near miss', 'almost', 'could have', 'potential', 'line of fire', 'energized', 'fall from height', 'suspended load', 'confined space']
    if severity_hint == 'high' and any(x in t for x in high_exposure_words):
        return 'high'
    if severity_hint == 'high':
        return 'high'
    if as_bool(row.get('any_injury')):
        return 'medium'
    if as_bool(row.get('is_overdue_task')):
        return 'medium'
    if any(x in t for x in ['near miss', 'almost', 'could have', 'potential', 'exposed', 'exposure']):
        return 'medium'
    return severity_hint or 'low'


def recommended_actions(pattern: str, row: pd.Series, taxonomy: Dict[str, Any]) -> List[str]:
    fam = get_family(taxonomy, pattern)
    actions = list(fam.get('actions', []))
    if not actions:
        actions = [
            'Review the event with the responsible supervisor and site EHS.',
            'Identify the failed, weak, or missing control and assign a corrective action.',
            'Check whether similar issues exist in the same area, department, or source system.'
        ]
    if as_bool(row.get('is_overdue_task')):
        actions.append('Escalate the overdue corrective action and confirm a realistic completion date.')
    if safe_str(row.get('source_type')).lower() == 'audit':
        actions.append('Use the audit finding to verify whether the same control gap exists in similar areas.')
    return actions[:5]


def summarize_event(row: pd.Series, text: str) -> str:
    title = compact_space(row.get('title'))
    if title and title.lower() not in {'title', 'description'}:
        return title[:260]
    return compact_space(text)[:260]


def evidence_phrases(text: str, tags: List[str]) -> List[str]:
    out = []
    low = text.lower()
    for tag in tags:
        tag_low = safe_str(tag).lower()
        if not tag_low:
            continue
        idx = low.find(tag_low)
        if idx >= 0:
            start = max(0, idx - 45)
            end = min(len(text), idx + len(tag_low) + 45)
            out.append(compact_space(text[start:end]))
    if not out and text:
        sents = re.split(r'(?<=[.!?])\s+', compact_space(text))
        out = [sents[0][:180]] if sents else [compact_space(text)[:180]]
    return out[:4]


def make_prompt(row: pd.Series) -> str:
    fields = {
        'event_id': safe_str(row.get('event_id')),
        'source_type': safe_str(row.get('source_type')),
        'source_subtype': safe_str(row.get('source_subtype')),
        'event_date': safe_str(row.get('event_date')),
        'site': safe_str(row.get('site')),
        'department': safe_str(row.get('department')),
        'category': safe_str(row.get('category')),
        'status': safe_str(row.get('status')),
        'title': safe_str(row.get('title')),
        'description': safe_str(row.get('description')),
        'clean_text': safe_str(row.get('clean_text')),
        'any_injury': safe_str(row.get('any_injury')),
        'severe_actual': safe_str(row.get('severe_actual')),
        'is_open_task': safe_str(row.get('is_open_task')),
        'is_overdue_task': safe_str(row.get('is_overdue_task')),
    }
    body = '\n'.join(f'{k}: {v}' for k, v in fields.items() if v)
    return (
        'Analyze this safety record and return only valid JSON using the requested schema.\n\n'
        'Requested JSON keys: ' + ', '.join(BASE_SCHEMA_KEYS) + '\n\n'
        'Safety record:\n' + body
    )


def make_output(row: pd.Series, taxonomy: Dict[str, Any], cfg: Dict[str, Any], discovered_theme: str | None = None) -> Dict[str, Any]:
    text = compact_space(' '.join([safe_str(row.get('title')), safe_str(row.get('description')), safe_str(row.get('clean_text'))]))
    source_type = safe_str(row.get('source_type')).lower()
    pattern, tags, consequence, pattern_score = infer_pattern(text, source_type, taxonomy, cfg)
    controls = infer_controls(text, taxonomy)
    risk = infer_risk_level(row, pattern, text, taxonomy)
    review_groups = ['Site EHS']
    if risk in {'high', 'critical'}:
        review_groups.append('Department Manager')
    if source_type == 'task':
        review_groups.append('Task Owner')
    if source_type == 'audit':
        review_groups.append('Audit Owner')

    limitations = []
    if pattern_score == 0:
        limitations.append('The event text did not strongly match a configured hazard family; classification should be reviewed.')
    if not safe_str(row.get('description')) and not safe_str(row.get('clean_text')):
        limitations.append('The record has limited narrative detail.')
    if not limitations:
        limitations.append('The output is based on event text and metadata only; it does not confirm root cause or control effectiveness.')

    return {
        'event_summary': summarize_event(row, text),
        'risk_pattern': pattern,
        'risk_pattern_description': (
            f"Primary weak pattern inferred from configured taxonomy/source-type logic: {pattern}. "
            "Light LLM labeling may add related or newly detected patterns in additional_patterns."
        ),
        'additional_patterns': [],
        'discovered_theme': discovered_theme or 'not clustered',
        'hazard_tags': tags[:8],
        'control_failure_tags': controls,
        'potential_consequence': consequence,
        'risk_level': risk,
        'recommended_actions': recommended_actions(pattern, row, taxonomy),
        'escalation_recommended': risk in {'high', 'critical'} or as_bool(row.get('is_overdue_task')),
        'recommended_review_group': sorted(set(review_groups)),
        'evidence_phrases': evidence_phrases(text, tags),
        'limitations': ' '.join(limitations)
    }


def create_chat_record(row: pd.Series, output: Dict[str, Any], label_source: str) -> Dict[str, Any]:
    return {
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': make_prompt(row)},
            {'role': 'assistant', 'content': json.dumps(output, ensure_ascii=False, indent=2)}
        ],
        'metadata': {
            'event_id': safe_str(row.get('event_id')),
            'source_type': safe_str(row.get('source_type')),
            'source_subtype': safe_str(row.get('source_subtype')),
            # Include these fields so downstream inference/demo sampling can report
            # site and department without having to re-parse the prompt text.
            'site': safe_str(row.get('site')),
            'department': safe_str(row.get('department')),
            'event_date': safe_str(row.get('event_date')),
            'category': safe_str(row.get('category')),
            'status': safe_str(row.get('status')),
            'label_source': label_source,
            'risk_pattern': output.get('risk_pattern'),
            'discovered_theme': output.get('discovered_theme'),
            'risk_level': output.get('risk_level')
        }
    }


def build_model_text(df: pd.DataFrame, taxonomy: Dict[str, Any], max_chars: int) -> pd.DataFrame:
    for col in ['event_id', 'source_type', 'source_subtype', 'title', 'description', 'clean_text']:
        if col not in df.columns:
            df[col] = ''
    text = (
        df['title'].fillna('').astype(str) + ' ' +
        df['description'].fillna('').astype(str) + ' ' +
        df['clean_text'].fillna('').astype(str)
    )
    df = df.copy()
    df['model_text'] = text.map(compact_space).str.slice(0, max_chars)
    # Do not normalize every row while scanning a large gzip CSV; it is expensive on CPU.
    # Normalized text is created only after the final sample is selected.
    return df


def load_and_sample(cfg: Dict[str, Any], taxonomy: Dict[str, Any]) -> pd.DataFrame:
    seed = int(cfg.get('seed', 42))
    sampling = cfg['sampling']
    out_dir = cfg_path(cfg, 'paths.prepared_dir')
    sampled_path = out_dir / 'sampled_records.csv'
    if sampling.get('reuse_existing_sampled_records', False) and sampled_path.exists():
        print(f'Reusing existing sampled records: {sampled_path}')
        return pd.read_csv(sampled_path)

    input_path = cfg_path(cfg, 'paths.input_csv')
    if not input_path.exists():
        raise FileNotFoundError(f'Input CSV not found: {input_path}')

    max_rows = sampling.get('max_rows_read')
    max_rows = None if max_rows in [None, 'null', 'None'] else int(max_rows)
    chunksize = int(sampling.get('chunksize', 50000))
    min_len = int(sampling.get('min_text_length', 80))
    max_chars = int(sampling.get('max_text_chars', 2200))
    targets = sampling.get('source_type_targets', {}) or {}
    candidate_parts: Dict[str, List[pd.DataFrame]] = defaultdict(list)
    seen = 0

    for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False):
        seen += len(chunk)
        chunk = build_model_text(chunk, taxonomy, max_chars=max_chars)
        chunk = chunk[chunk['model_text'].str.len() >= min_len].copy()
        if sampling.get('prefer_english_only', False) and 'language_hint' in chunk.columns:
            chunk = chunk[chunk['language_hint'].fillna('').str.lower().isin(['en', 'english', ''])]
        if not len(chunk):
            if max_rows is not None and seen >= max_rows:
                break
            continue
        if targets:
            for source_type, target_n in targets.items():
                sub = chunk[chunk['source_type'].fillna('').str.lower() == source_type.lower()]
                if len(sub):
                    take = min(len(sub), max(int(target_n) * 2, 100))
                    candidate_parts[source_type].append(sub.sample(n=take, random_state=seed + len(candidate_parts[source_type])))
        else:
            candidate_parts['all'].append(chunk.sample(n=min(len(chunk), 2000), random_state=seed))
        if max_rows is not None and seen >= max_rows:
            break

    samples = []
    if targets:
        for source_type, target_n in targets.items():
            parts = candidate_parts.get(source_type, [])
            if not parts:
                continue
            pool = pd.concat(parts, ignore_index=True)
            dedupe_col = 'model_text'
            pool = pool.drop_duplicates(subset=[dedupe_col])
            samples.append(pool.sample(n=min(int(target_n), len(pool)), random_state=seed))
    else:
        pool = pd.concat(candidate_parts['all'], ignore_index=True).drop_duplicates(subset=['model_text'])
        samples.append(pool.sample(n=min(int(sampling.get('max_examples_total', 3000)), len(pool)), random_state=seed))
    if not samples:
        raise ValueError('No usable safety text records were sampled. Check input path and filters.')
    out = pd.concat(samples, ignore_index=True)
    max_total = int(sampling.get('max_examples_total', 3000))
    if len(out) > max_total:
        out = out.sample(n=max_total, random_state=seed)
    if sampling.get('dedupe_on_normalized_text', True):
        out['normalized_text'] = out['model_text'].map(lambda x: normalize_text(x, taxonomy))
        out = out.drop_duplicates(subset=['normalized_text'])
        if len(out) > max_total:
            out = out.sample(n=max_total, random_state=seed)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def label_from_terms(terms: List[str], stopwords: set[str], max_terms: int = 4) -> str:
    cleaned = []
    for term in terms:
        words = [w for w in term.split() if w not in stopwords and len(w) >= 3]
        if not words:
            continue
        phrase = ' '.join(words)
        if phrase not in cleaned:
            cleaned.append(phrase)
        if len(cleaned) >= max_terms:
            break
    return ' / '.join(cleaned) if cleaned else 'mixed safety records'


def cluster_one_frame(df: pd.DataFrame, cfg: Dict[str, Any], taxonomy: Dict[str, Any], source_label: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    cl_cfg = cfg['labeling']['clustering']
    stopwords = build_stopwords(taxonomy)
    tokenizer = tokenizer_factory(stopwords, use_stemming=bool(cl_cfg.get('use_stemming', True)))
    ngram_range = tuple(cl_cfg.get('ngram_range', [1, 3]))
    max_features = int(cl_cfg.get('max_features', 18000))
    min_df = int(cl_cfg.get('min_df', 3))
    max_df = float(cl_cfg.get('max_df', 0.65))
    raw_texts = df['model_text'].fillna('').astype(str).tolist()
    # Pre-tokenize once, then let scikit-learn build word n-grams on cleaned text.
    # This is much faster on CPU than using a Python tokenizer inside TfidfVectorizer for every n-gram pass.
    texts = [' '.join(tokenizer(t)) for t in raw_texts]
    keep_mask = [len(t.split()) >= 3 for t in texts]
    if sum(keep_mask) < 8:
        return pd.DataFrame(), {safe_str(e): 'not enough informative terms to cluster' for e in df['event_id']}
    if not all(keep_mask):
        df = df.loc[keep_mask].reset_index(drop=True)
        texts = [t for t, keep in zip(texts, keep_mask) if keep]

    vectorizer = TfidfVectorizer(
        lowercase=False,
        token_pattern=r'(?u)\b\w[\w\-]+\b',
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        norm='l2'
    )
    try:
        X = vectorizer.fit_transform(texts)
    except ValueError as ex:
        print(f'WARNING: clustering skipped for {source_label}: {ex}')
        return pd.DataFrame(), {safe_str(e): 'clustering skipped' for e in df['event_id']}

    by_src = cl_cfg.get('n_clusters_by_source_type', {}) or {}
    desired = int(by_src.get(source_label, cl_cfg.get('n_clusters', 12)))
    n_clusters = max(2, min(desired, max(2, len(df) // 20), len(df) - 1))
    if X.shape[1] > 80 and X.shape[0] > 30:
        n_comp = max(2, min(50, X.shape[1] - 1, X.shape[0] - 1))
        X_cluster = TruncatedSVD(n_components=n_comp, random_state=int(cfg.get('seed', 42))).fit_transform(X)
    else:
        X_cluster = X
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=int(cfg.get('seed', 42)),
        batch_size=1024,
        n_init=1,
        max_iter=60,
        reassignment_ratio=0.0
    )
    labels = km.fit_predict(X_cluster)
    terms = np.array(vectorizer.get_feature_names_out())
    top_n = int(cl_cfg.get('top_terms_per_cluster', 10))
    min_size = int(cl_cfg.get('min_cluster_size_for_label', 5))
    summary_rows = []
    event_theme = {}

    # Silhouette can be expensive; sample to keep CPU acceptable.
    sil = None  # disabled by default for CPU speed; enable manually if cluster validation is needed

    for cid in range(n_clusters):
        idx = np.where(labels == cid)[0]
        size = len(idx)
        # Use the mean original TF-IDF weights within the cluster to label the cluster, even when
        # clustering is performed on SVD-reduced vectors.
        center = np.asarray(X[idx].mean(axis=0)).ravel()
        top_idx = center.argsort()[::-1]
        raw_terms = [terms[i] for i in top_idx if center[i] > 0][:top_n * 3]
        # Remove any remaining bad terms from labels.
        label_terms = []
        for t in raw_terms:
            if any(w in stopwords for w in t.split()):
                continue
            label_terms.append(t)
            if len(label_terms) >= top_n:
                break
        cluster_label = label_from_terms(label_terms, stopwords)
        if size < min_size:
            cluster_label = f'small cluster: {cluster_label}'
        full_label = f'{source_label}: {cluster_label}' if source_label != 'all' else cluster_label
        summary_rows.append({
            'source_type': source_label,
            'cluster_id': f'{source_label}_{cid}',
            'cluster_label': full_label,
            'size': int(size),
            'top_terms': '|'.join(label_terms[:top_n]),
            'silhouette_cosine_sample': sil
        })
        for i in idx:
            event_theme[safe_str(df.iloc[i].get('event_id'))] = full_label
    return pd.DataFrame(summary_rows), event_theme


def run_clustering(df: pd.DataFrame, cfg: Dict[str, Any], taxonomy: Dict[str, Any], out_dir: Path) -> Dict[str, str]:
    cl_cfg = cfg.get('labeling', {}).get('clustering', {})
    if not cl_cfg.get('enabled', True):
        return {}
    max_records = int(cl_cfg.get('max_records', 10000))
    seed = int(cfg.get('seed', 42))
    work = df.sample(n=min(max_records, len(df)), random_state=seed).copy()
    summaries = []
    theme_map = {}
    if cl_cfg.get('cluster_by_source_type', True):
        for source_type, sub in work.groupby(work['source_type'].fillna('unknown').str.lower()):
            print(f'  clustering source_type={source_type}, n={len(sub)}', flush=True)
            summary, mapping = cluster_one_frame(sub.reset_index(drop=True), cfg, taxonomy, source_type)
            print(f'  finished source_type={source_type}', flush=True)
            if len(summary):
                summaries.append(summary)
            theme_map.update(mapping)
    else:
        summary, mapping = cluster_one_frame(work.reset_index(drop=True), cfg, taxonomy, 'all')
        if len(summary):
            summaries.append(summary)
        theme_map.update(mapping)
    if summaries:
        summary_df = pd.concat(summaries, ignore_index=True).sort_values(['source_type', 'size'], ascending=[True, False])
    else:
        summary_df = pd.DataFrame(columns=['source_type', 'cluster_id', 'cluster_label', 'size', 'top_terms'])
    summary_df.to_csv(out_dir / 'cluster_summary.csv', index=False)
    return theme_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config.yaml')
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get('seed', 42)))
    taxonomy = load_yaml(cfg_path(cfg, 'paths.taxonomy_file'))
    out_dir = ensure_dir(cfg_path(cfg, 'paths.prepared_dir'))

    print('Loading and sampling records...', flush=True)
    df = load_and_sample(cfg, taxonomy)
    print(f'Sampled {len(df)} records', flush=True)
    df.to_csv(out_dir / 'sampled_records.csv', index=False)
    print('Running clustering...', flush=True)
    cluster_map = run_clustering(df, cfg, taxonomy, out_dir)
    print(f'Cluster theme map contains {len(cluster_map)} records', flush=True)

    records = []
    print('Creating chat records...', flush=True)
    for i, row in enumerate(df.to_dict(orient='records')):
        if i % 250 == 0:
            print(f'  created {i} records', flush=True)
        eid = safe_str(row.get('event_id'))
        discovered = cluster_map.get(eid, 'not clustered')
        output = make_output(row, taxonomy, cfg, discovered_theme=discovered)
        records.append(create_chat_record(row, output, label_source='taxonomy_plus_cleaned_cluster'))

    n = len(records)
    train_n = int(n * float(cfg['sampling'].get('train_ratio', 0.8)))
    val_n = int(n * float(cfg['sampling'].get('val_ratio', 0.1)))
    train = records[:train_n]
    val = records[train_n:train_n + val_n]
    test = records[train_n + val_n:]
    print('Writing JSONL files...', flush=True)
    write_jsonl(train, out_dir / 'safety_instruction_train.jsonl')
    write_jsonl(val, out_dir / 'safety_instruction_val.jsonl')
    write_jsonl(test, out_dir / 'safety_instruction_test.jsonl')

    manifest = {
        'n_total': n,
        'n_train': len(train),
        'n_val': len(val),
        'n_test': len(test),
        'source_type_counts': df['source_type'].value_counts(dropna=False).to_dict(),
        'label_source': 'taxonomy_plus_cleaned_cluster',
        'taxonomy_file': str(cfg_path(cfg, 'paths.taxonomy_file')),
        'prepared_dir': str(out_dir),
        'note': 'Risk labels are weak labels from editable taxonomy plus improved TF-IDF clustering. SME review is still required before production use.'
    }
    with open(out_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
