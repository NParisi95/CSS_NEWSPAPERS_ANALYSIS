import pandas as pd

import os

from THEME_DICTIONARY import themes
    
fp_only = os.path.join(os.getcwd(), r"ARTICLES DATA\MERGED DATASET\merged_output_TRANSLATED_combined.csv")


df = pd.read_csv(fp_only)
df = df.dropna(subset=['Text_lemm'])

def determine_article_theme(text, themes):
    # Calcola i punteggi per ogni categoria
    scores = {theme: 0.0 for theme in themes}
    words = text.split()
    for word in words:
        for theme, keywords in themes.items():
            for keyword, weight in keywords:
                if word == keyword:
                    scores[theme] += weight

    # Controlla se tutti i punteggi sono zero
    if all(score == 0 for score in scores.values()):
        return "other"
    
    highest_score_theme = max(scores, key=scores.get)
    return highest_score_theme

# Funzione per assegnare il tema agli articoli nel DataFrame
def assign_themes_to_articles(df, themes):
    df['type'] = df['Text_lemm'].apply(lambda text: determine_article_theme(text, themes))

# Esempio di utilizzo
assign_themes_to_articles(df, themes)

df.to_csv(fp_only)