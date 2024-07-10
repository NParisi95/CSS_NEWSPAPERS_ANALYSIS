# ================================
# CLOROPLETH MAP
# ================================

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

regions = gpd.read_file(os.path.join(current_dir, 'limits_IT_regions.geojson'))

csv_path = os.path.join(current_dir, 'ARTICLES DATA', 'ANSA.csv')
polarity_scores = pd.read_csv(csv_path)

polarity_scores['Source'] = polarity_scores['Source'].str.lower()

mean_polarity_scores = polarity_scores.groupby('Source')['polarity score'].mean().reset_index()
print(mean_polarity_scores)

regions['reg_name'] = regions['reg_name'].str.lower()

regions = regions.merge(mean_polarity_scores, how='left', left_on='reg_name', right_on='Source')

fig, ax = plt.subplots(1, 1, figsize=(15, 10))
regions.boundary.plot(ax=ax, linewidth=1, edgecolor='black')
regions.plot(column='polarity score', cmap='coolwarm', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True, missing_kwds={'color': 'white', 'label': 'Missing values'})

for idx, row in regions.iterrows():
    if pd.notnull(row['polarity score']):
        plt.annotate(s=round(row['polarity score'], 2), xy=(row['geometry'].centroid.x, row['geometry'].centroid.y),
                     horizontalalignment='center', fontsize=8, color='black')

plt.title('Average Polarity Scores by Italian Region')
plt.show()


# ===============================
# THEME PLOT
# ===============================

import pandas as pd
import matplotlib.pyplot as plt
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(current_dir, 'ARTICLES DATA', 'ANSA.csv')
df = pd.read_csv(csv_path)

region_genre_counts = df.groupby(['Source', 'type']).size().reset_index(name='counts')

region_genre_counts = region_genre_counts.sort_values(['Source', 'counts'], ascending=[True, False])

top_genres_per_region = region_genre_counts.groupby('Source').head(5)

pivot_df = top_genres_per_region.pivot(index='Source', columns='type', values='counts').fillna(0)

pivot_df.plot(kind='bar', stacked=True, figsize=(10, 15))

plt.title('Top 5 Themes by Region')
plt.xlabel('Region')
plt.ylabel('Count')
plt.legend(title='Theme')
plt.xticks(rotation=90)
plt.tight_layout()

plt.show()


