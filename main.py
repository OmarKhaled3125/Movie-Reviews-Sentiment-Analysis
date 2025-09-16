
"""
Movie Review Sentiment Analysis — Compare 4 Algorithms
"""

import nltk
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score
import csv
from pathlib import Path

# --- Ensure NLTK dataset ---
def ensure_nltk_resources():
    try:
        nltk.data.find("corpora/movie_reviews")
    except LookupError:
        nltk.download("movie_reviews")

# --- Load dataset ---
def load_dataset():
    ensure_nltk_resources()
    docs, labels = [], []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            docs.append(movie_reviews.raw(fileid))
            labels.append(category)
    return docs, labels

# --- Main ---
def main():
    print("Loading dataset...")
    X, y = load_dataset()

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Common TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000)

    results = []

    # --- Naive Bayes ---
    nb_pipe = Pipeline([
        ("tfidf", vectorizer),
        ("clf", MultinomialNB()),
    ])
    print("\nTraining Naive Bayes...")
    nb_pipe.fit(X_train, y_train)
    nb_pred = nb_pipe.predict(X_test)
    nb_acc = accuracy_score(y_test, nb_pred)
    nb_prec = precision_score(y_test, nb_pred, average="weighted", zero_division=0)
    nb_f1 = f1_score(y_test, nb_pred, average="weighted", zero_division=0)
    results.append(("Naive Bayes", nb_acc, nb_prec, nb_f1))

    # --- Logistic Regression ---
    logreg_pipe = Pipeline([
        ("tfidf", vectorizer),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    print("\nTraining Logistic Regression...")
    logreg_pipe.fit(X_train, y_train)
    logreg_pred = logreg_pipe.predict(X_test)
    logreg_acc = accuracy_score(y_test, logreg_pred)
    logreg_prec = precision_score(y_test, logreg_pred, average="weighted", zero_division=0)
    logreg_f1 = f1_score(y_test, logreg_pred, average="weighted", zero_division=0)
    results.append(("Logistic Regression", logreg_acc, logreg_prec, logreg_f1))

    # --- Linear SVM ---
    svm_pipe = Pipeline([
        ("tfidf", vectorizer),
        ("clf", LinearSVC(dual=False)),
    ])
    print("\nTraining Linear SVM...")
    svm_pipe.fit(X_train, y_train)
    svm_pred = svm_pipe.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_pred)
    svm_prec = precision_score(y_test, svm_pred, average="weighted", zero_division=0)
    svm_f1 = f1_score(y_test, svm_pred, average="weighted", zero_division=0)
    results.append(("Linear SVM", svm_acc, svm_prec, svm_f1))

    # --- Random Forest ---
    rf_pipe = Pipeline([
        ("tfidf", vectorizer),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42)),
    ])
    print("\nTraining Random Forest...")
    rf_pipe.fit(X_train, y_train)
    rf_pred = rf_pipe.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_prec = precision_score(y_test, rf_pred, average="weighted", zero_division=0)
    rf_f1 = f1_score(y_test, rf_pred, average="weighted", zero_division=0)
    results.append(("Random Forest", rf_acc, rf_prec, rf_f1))

    # --- Print results ---
    print("\n=== Model Comparison ===")
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'F1-score':<10}")
    print("-" * 55)
    for name, acc, prec, f1 in results:
        print(f"{name:<20} {acc:.4f}    {prec:.4f}    {f1:.4f}")

    # --- Save results to CSV ---
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "model_comparison.csv"

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Accuracy", "Precision", "F1-score"])
        for name, acc, prec, f1 in results:
            writer.writerow([name, f"{acc:.4f}", f"{prec:.4f}", f"{f1:.4f}"])

    print(f"\nResults saved to: {csv_path.resolve()}")

if __name__ == "__main__":
    main()
