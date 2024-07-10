# =======================================================
# ARTICLES FREQUENCY
# =======================================================

import os
import pandas as pd
from THEME_DICTIONARY import themes
import matplotlib.pyplot as plt
PATH = os.path.join(os.getcwd(), "ARTICLES DATA/MERGED DATASET/merged_output_TRANSLATED_combined.csv")

ofp_df = path + r"\merged_output_FIRST_PAGE_ONLY.csv"

df = pd.read_csv(ofp_df)

article_counts = df["testata"].value_counts()

plt.figure(figsize=(10, 5))
article_counts.plot(kind='bar')

# Ruota le etichette delle x in modo che siano verticali
plt.xticks(rotation=90)

# Titolo e nomi degli assi
plt.title('Number of Articles Published by Newspaper')
plt.xlabel('Newspaper')
plt.ylabel('Number of Articles')

# Salva l'immagine in alta definizione
plt.savefig(r'C:\Users\npari\Desktop\nik\DATA SCIENCE\computational social science\PAPER\PLOTS\articles_by_newspaper.png', dpi=300)

plt.tight_layout()
plt.show()

