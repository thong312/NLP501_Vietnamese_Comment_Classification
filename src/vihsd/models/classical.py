from vihsd.data import normalize


def make_pipeline(c=1.0):
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2),
                                 min_df=2, max_features=80000, sublinear_tf=True)),
        ("classifier", LogisticRegression(C=c, class_weight="balanced", max_iter=1000, random_state=42)),
    ])
