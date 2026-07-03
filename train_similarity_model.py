"""
train_similarity_model.py
--------------------------
Run this whenever data/toxin_phases.csv changes:
    python train_similarity_model.py

Trains on 25 toxins x timeline phases (72 rows) from the client's
Forensic_Toxicology_AI_DSS dataset. Saves model/toxin_matcher.pkl,
loaded by app.py at runtime.
"""

from code.toxin_matcher import ToxinMatcher

matcher = ToxinMatcher.train_from_csv("data/toxin_phases.csv")
matcher.save("model/toxin_matcher.pkl")

print("Saved model/toxin_matcher.pkl")
print(f"Trained on {len(matcher.toxin_labels)} phase-rows, "
      f"{len(set(matcher.toxin_labels))} distinct toxins.")
