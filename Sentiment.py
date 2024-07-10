# =======================================================
# TRANSLATION
# =======================================================

import os
import pandas as pd
from deep_translator import GoogleTranslator

fp_only = os.path.join(os.getcwd(), "ARTICLES DATA/MERGED DATASET/merged_output_FIRST_PAGE_ONLY.csv")

df = pd.read_csv(fp_only)

df['Translation'] = None
translated_path = os.path.join(os.getcwd(), "ARTICLES DATA/MERGED DATASET/merged_output_TRANSLATED.csv")
translated_df = pd.read_csv(translated_path)

df = df.iloc[len(translated_df):]

counter = len(df)

for index, row in df.iterrows():
    try:
        translated_text = GoogleTranslator(source='it', target='en').translate(row['Summary'])
        df.at[index, 'Translation'] = translated_text
    except Exception as e:
        print(f"Error translating row {index}: {e}")
    if counter % 30 == 0:
        print(counter)
    
    counter -= 1

output_fp = os.path.join(os.getcwd(), "ARTICLES DATA/MERGED DATASET/merged_output_TRANSLATED.csv")
df.to_csv(output_fp, index=False)
print(f"Translated file saved as {output_fp}")



# =======================================================
# SENTIMENT ANALYSIS (VADER)
# =======================================================

import os
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

path = os.path.join(os.getcwd(), "ARTICLES DATA/MERGED DATASET/merged_output_TRANSLATED_combined.csv")

df = pd.read_csv(path)

def calculate_vader_polarity(text):
    scores = sia.polarity_scores(text)
    return scores['compound']

df['polarity_score'] = df['Translation'].apply(calculate_vader_polarity)

df.to_csv(path, index=False)
print("Polarity score added successfully!")



# =======================================================
# VISUALIZATION (Polarity Distribution)
# =======================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = os.path.join(os.getcwd(), "ARTICLES DATA/MERGED DATASET/merged_output_TRANSLATED_combined.csv")
df = pd.read_csv(path)

plt.figure(figsize=(15, 10))
sns.boxplot(x='newspaper', y='polarity_score', data=df)

plt.xticks(rotation=90)
plt.title('Polarity distribution')
plt.xlabel('Newspaper')
plt.ylabel('Polarity Score')

plt.tight_layout()
plt.show()

for date, group in df.groupby("published_date"):
    print(date, " >>", group["polarity_score"].mean())



# =======================================================
# VISUALIZATION (Article Genre Proportions)
# =======================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = os.path.join(os.getcwd(), "ARTICLES DATA/MERGED DATASET/merged_output_TRANSLATED_combined.csv")
df = pd.read_csv(path)

pivot_df = df.pivot_table(index='newspaper', columns='genre', aggfunc='size', fill_value=0)

proportions_df = pivot_df.div(pivot_df.sum(axis=1), axis=0)

num_categories = len(proportions_df.columns)
colors = plt.cm.get_cmap('tab20b', num_categories)

plt.figure(figsize=(20, 10))
proportions_df.plot(kind='bar', stacked=True, figsize=(15, 10), colormap=colors)

plt.title('Normalized distribution of article genres per publication')
plt.xlabel('Publication')
plt.ylabel('Proportion of articles')

plt.xticks(rotation=90)
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
