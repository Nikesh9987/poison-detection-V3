"""
toxin_matcher.py
-----------------
Symptom-to-toxin matcher for VishAI.

WHY THIS EXISTS (read before touching):
The original approach trained a RandomForestClassifier on TF-IDF vectors
built from only 27 rows across 10 toxin classes (2-3 samples each). With
that little data, RandomForest cannot generalise -- it just memorises the
training rows and, for any input that doesn't look like one of those exact
paragraphs, falls back to whichever class dominates the leaf nodes nearest
the origin. That's why every symptom combination (even an empty string)
predicted "Copper Sulphate".

This module replaces the classifier with a similarity-search approach:
we TF-IDF-vectorise the training text as before, but instead of asking a
classifier to draw decision boundaries from 27 points, we just find which
training document(s) the input is *most similar to* (cosine similarity).
This degrades gracefully with tiny datasets and gives a real confidence
score you can threshold on -- exactly what you need until the dataset is
bigger (see train_similarity_model.py + README notes for scaling this up).
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Frontend/


class ToxinMatcher:
    def __init__(self, vectorizer, doc_vectors, toxin_labels):
        self.vectorizer = vectorizer
        self.doc_vectors = doc_vectors
        self.toxin_labels = toxin_labels  # list, same order as doc_vectors rows

    def predict(self, symptoms_text: str, min_confidence: float = 0.05):
        """
        Returns (best_label, confidence, ranked_list)
        - best_label: str or None if nothing scored above min_confidence
        - confidence: float 0..1 (cosine similarity of the best match)
        - ranked_list: list of (label, score) sorted descending, deduped by toxin
        """
        query_vec = self.vectorizer.transform([symptoms_text])
        sims = cosine_similarity(query_vec, self.doc_vectors)[0]

        # a toxin can have multiple rows (multiple severity stages) -- take its best row
        best_per_toxin = {}
        for label, score in zip(self.toxin_labels, sims):
            if label not in best_per_toxin or score > best_per_toxin[label]:
                best_per_toxin[label] = score

        ranked = sorted(best_per_toxin.items(), key=lambda x: -x[1])

        if not ranked or ranked[0][1] < min_confidence:
            return None, 0.0, ranked

        return ranked[0][0], float(ranked[0][1]), ranked

    def save(self, path):
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "doc_vectors": self.doc_vectors,
                "toxin_labels": self.toxin_labels,
            },
            path,
        )

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        return cls(data["vectorizer"], data["doc_vectors"], data["toxin_labels"])

    @classmethod
    def train_from_csv(cls, csv_path):
        """
        Expects data/toxin_phases.csv with columns:
        toxin_id, toxin_name, category, phase, cns, cvs, respiratory, gi, other, odour, route, sample
        One row per (toxin, timeline phase) -- this is the granular training unit,
        already vernacular-cleaned (English/scientific names only).
        """
        df = pd.read_csv(csv_path).fillna("")
        df["features"] = (
            df["cns"].astype(str) + " " +
            df["cvs"].astype(str) + " " +
            df["respiratory"].astype(str) + " " +
            df["gi"].astype(str) + " " +
            df["other"].astype(str) + " " +
            df["odour"].astype(str)
        )
        toxin_labels = df["toxin_name"].astype(str).str.strip().tolist()

        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 1),
            lowercase=True,
            stop_words="english",
        )
        doc_vectors = vectorizer.fit_transform(df["features"])

        return cls(vectorizer, doc_vectors, toxin_labels)
