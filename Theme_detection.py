# =======================================================
# THEME ASSIGNMENT
# =======================================================

import os
import pandas as pd
from THEME_DICTIONARY import themes

fp_only = os.path.join(os.getcwd(), "ARTICLES DATA/MERGED DATASET/merged_output_TRANSLATED_combined.csv")

df = pd.read_csv(fp_only)
df = df.dropna(subset=['Text_lemm'])

def determine_article_theme(text, themes):
    scores = {theme: 0.0 for theme in themes}
    words = text.split()
    for word in words:
        for theme, keywords in themes.items():
            for keyword, weight in keywords:
                if word == keyword:
                    scores[theme] += weight
    if all(score == 0 for score in scores.values()):
        return "other"
    highest_score_theme = max(scores, key=scores.get)
    return highest_score_theme

def assign_themes_to_articles(df, themes):
    df['theme'] = df['Text_lemm'].apply(lambda text: determine_article_theme(text, themes))

assign_themes_to_articles(df, themes)

df.to_csv(fp_only, index=False)
print(f"Themes assigned and saved in {fp_only}")
