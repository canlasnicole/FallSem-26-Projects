# ============================================================
# Part 3 확장 실험: Sentence-BERT 임베딩 vs TF-IDF (단독 실행 스크립트, 로컬/Colab 겸용)
# ------------------------------------------------------------
# 사용법 (Google Colab)
#   1) 왼쪽 파일 패널에 starwars_clean_small_125.csv, starwars_clean_large_4char.csv 업로드
#   2) 새 셀에 아래 한 줄 실행:   !pip install -q sentence-transformers
#   3) 이 파일 전체를 다음 셀에 붙여넣고 실행 (CPU 런타임으로 충분, 약 2~5분)
#   4) 마지막에 출력되는 표를 복사해 보고서 Q5에 반영
#
# 메인 노트북과 동일한 조건: RepeatedStratifiedKFold(5-Fold × 5회, random_state=42),
# 동일 지표(Accuracy, Bad P/R/F1, Macro-F1), 동일 분류기(LogReg, LinearSVC, none/balanced).
# SBERT 인코더는 학습되지 않는 고정 특징 추출기이므로 전체 문장을 한 번 인코딩해도
# 라벨 누출이 없다. 분류기만 각 훈련 폴드에서 학습한다.
# ============================================================
import time, platform
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score

SEED = 42
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
import os
DATA_DIR = "data" if os.path.isdir("data") else "."   # 리포지토리에서는 data/, Colab 업로드 시 현재 폴더
FILES = {"Small": os.path.join(DATA_DIR, "starwars_clean_small_125.csv"),
         "Large": os.path.join(DATA_DIR, "starwars_clean_large_4char.csv")}
OUT = "results/sbert_vs_tfidf_results.csv" if os.path.isdir("results") else "sbert_vs_tfidf_results.csv"


def metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0], zero_division=0)
    return {"Accuracy": accuracy_score(y_true, y_pred), "Bad Precision": p[0], "Bad Recall": r[0],
            "Bad F1": f[0], "Macro-F1": f1_score(y_true, y_pred, average="macro", zero_division=0)}


def make_clf(name, balanced):
    cw = "balanced" if balanced else None
    if name == "LogReg":
        return LogisticRegression(max_iter=2000, class_weight=cw, random_state=SEED)
    return LinearSVC(class_weight=cw, random_state=SEED)


def cv_eval(make_estimator, X, y):
    """X는 문장 배열(TF-IDF 파이프라인) 또는 임베딩 행렬(SBERT). 폴드별 지표 DataFrame 반환."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    rows = []
    for tr, te in cv.split(np.zeros(len(y)), y):
        est = make_estimator().fit(X[tr], y[tr])
        rows.append(metrics(y[te], est.predict(X[te])))
    return pd.DataFrame(rows)


def run(encoder=None):
    """encoder: 문장 리스트 → (n, d) 배열 함수. None이면 SentenceTransformer를 로드."""
    if encoder is None:
        from sentence_transformers import SentenceTransformer
        t0 = time.time()
        st = SentenceTransformer(MODEL_NAME, device="cpu")
        print(f"모델 로드: {time.time() - t0:.1f}s")
        encoder = lambda texts: st.encode(texts, batch_size=64, normalize_embeddings=True,
                                          show_progress_bar=False)
    results, timing = [], []
    for size, path in FILES.items():
        df = pd.read_csv(path, encoding="utf-8-sig")
        texts, y = df["Text"].astype(str).values, df["Target"].values

        t0 = time.time(); emb = np.asarray(encoder(list(texts))); enc_sec = time.time() - t0
        timing.append({"Dataset": f"{size} (n={len(y)})", "Embedding dim": emb.shape[1],
                       "Encoding time (s)": round(enc_sec, 2),
                       "ms / sentence": round(1000 * enc_sec / len(y), 2)})

        for feat in ["TF-IDF", "SBERT MiniLM-L6"]:
            for m in ["LogReg", "LinearSVC"]:
                for bal in [False, True]:
                    if feat == "TF-IDF":
                        X = texts
                        make = lambda: Pipeline([("vec", TfidfVectorizer(ngram_range=(1, 2), min_df=1,
                                                                           sublinear_tf=True)),
                                                 ("clf", make_clf(m, bal))])
                    else:
                        X = emb
                        make = lambda: make_clf(m, bal)
                    t0 = time.time(); folds = cv_eval(make, X, y); fit_sec = time.time() - t0
                    row = {"Dataset": f"{size} (n={len(y)})", "Features": feat, "Model": m,
                           "Imbalance": "balanced" if bal else "none"}
                    for c in folds.columns:
                        row[c] = f"{folds[c].mean():.3f} ± {folds[c].std():.3f}"
                    row["CV time (s)"] = round(fit_sec, 1)
                    results.append(row)

    res = pd.DataFrame(results)
    print("\n=== 인코딩 비용 ===")
    print(pd.DataFrame(timing).to_string(index=False))
    print(f"환경: {platform.platform()} | CPU 런타임 기준")
    print("\n=== 결과 (25개 폴드 평균 ± 표준편차) ===")
    print(res.to_string(index=False))
    res.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUT}")
    return res


if __name__ == "__main__":
    run()
