============================================
MEAN WORDS PER ARTICLE
============================================

import pandas as pd
import os
import matplotlib.pyplot as plt

# Load data
fp_only = os.path.join(os.getcwd(), "ARTICLES DATA", "MERGED DATASET", "merged_output_FIRST_PAGE_ONLY.csv")
df_fp_only = pd.read_csv(fp_only)

def mean_words(dfcleaned, column="Text", newspaper=None):
    total_words = 0
    
    if newspaper is None:
        df = dfcleaned[column]
    else:
        df = dfcleaned[column][dfcleaned["testata"] == newspaper]
        
    for el in df:
        el = el.split()
        total_words += len(el)
        
    return total_words // len(df)

def plot_mean_words(dfcleaned, name):
    mean_words_global = mean_words(dfcleaned)
    
    mean_words_dict = {}
    for el in dfcleaned["testata"].unique():
        mean_words_dict[el] = mean_words(dfcleaned, newspaper=el)
    
    sorted_mean_words = dict(sorted(mean_words_dict.items(), key=lambda item: item[1], reverse=True))
    
    # Create subplots with 2 rows and 1 column
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot first graph (original data)
    ax1.bar(sorted_mean_words.keys(), sorted_mean_words.values(), width=0.5, align='center')
    ax1.set_xlabel('Newspaper')
    ax1.set_ylabel('Average number of words')
    ax1.set_title('Average number of words per newspaper')
    ax1.grid(True)
    ax1.set_xticklabels(sorted_mean_words.keys(), rotation=90)
    ax1.axhline(y=mean_words_global, color='r', linestyle='-', label=f'Global average: {mean_words_global:.2f} words')
    ax1.legend()
    
    # Plot second graph (summarized data)
    mean_words_summary = mean_words(dfcleaned, column="Summary")
    
    mean_words_summary_dict = {}
    for el in dfcleaned["testata"].unique():
        mean_words_summary_dict[el] = mean_words(dfcleaned, column="Summary", newspaper=el)
    
    sorted_mean_words_summary = dict(sorted(mean_words_summary_dict.items(), key=lambda item: item[1], reverse=True))
    
    ax2.bar(sorted_mean_words_summary.keys(), sorted_mean_words_summary.values(), width=0.5, align='center')
    ax2.set_xlabel('Newspaper')
    ax2.set_ylabel('Average number of words')
    ax2.set_title('Average number of words per newspaper (SUMMARIZED ARTICLES)')
    ax2.grid(True)
    ax2.set_xticklabels(sorted_mean_words_summary.keys(), rotation=90)
    ax2.axhline(y=mean_words_summary, color='r', linestyle='-', label=f'Global average: {mean_words_summary:.2f} words')
    ax2.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the combined plot
    plot_output_path = os.path.join(os.getcwd(), "PLOTS", f"{name}.png")
    plt.savefig(plot_output_path, dpi=500, bbox_inches='tight')
    
    # Show the plot
    plt.show()

# Plot the combined graphs
plot_mean_words(df_fp_only, "mean_words_combined")
plot_mean_words(df_fp_only, "WORDS_COUNT")



========================================
GRAPH PLOT
========================================

import pandas as pd
import os
import networkx as nx
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
path = os.path.join(os.getcwd(), "ARTICLES DATA", "MERGED DATASET", "TF-IDF DATAFRAME", "TEXT_LEMM_top_words.csv")
df = pd.read_csv(path)

newspaper_lists = {col: df[col].tolist() for col in df.columns}

common_words = set(newspaper_lists[df.columns[0]])
for col in df.columns[1:]:
    common_words &= set(newspaper_lists[col])

filtered_lists = {}
for col in df.columns:
    filtered_lists[col] = [word for word in newspaper_lists[col] if word not in common_words]

final_lists = {col: filtered_lists[col][:100] for col in df.columns}

df_new = pd.DataFrame.from_dict(final_lists)
df = df_new

G = nx.Graph()

confusion_matrix = pd.DataFrame(0, index=df.columns, columns=df.columns)

for newspaper1, newspaper2 in combinations(df.columns, 2):
    common_words_between = set(df[newspaper1]).intersection(set(df[newspaper2])) - common_words
    confusion_matrix.loc[newspaper1, newspaper2] = len(common_words_between)
    confusion_matrix.loc[newspaper2, newspaper1] = len(common_words_between)

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="YlGnBu", cbar=False)
plt.title("Confusion Matrix of Common Words between Newspapers (Excluding Words Common to All)")
plt.xlabel("Newspapers")
plt.ylabel("Newspapers")
plt.show()

for col in df.columns:
    G.add_node(col)

for newspaper1, newspaper2 in combinations(df.columns, 2):
    weight = confusion_matrix.loc[newspaper1, newspaper2]
    if weight > 0:
        G.add_edge(newspaper1, newspaper2, weight=weight)

pos = nx.circular_layout(G)

edges = G.edges(data=True)
weights = [d['weight'] for (u, v, d) in edges]

norm = plt.Normalize(min(weights), max(weights))
cmap = plt.cm.Greys

plt.figure(figsize=(12, 10))

nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='skyblue')

for (u, v, d) in edges:
    weight = d['weight']
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=weight / 5, alpha=0.6, edge_color=cmap(norm(weight)))

labels = {node: node for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=14, font_color='black')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

plt.title("Visualization of Newspapers with Common Words")
plt.tight_layout()
plt.show()


===========================================
NEWS TREND
===========================================

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load data
path = os.path.join(os.getcwd(), "ARTICLES DATA", "MERGED DATASET", "TF-IDF DATAFRAME", "DATE", "TEXT_LEMM_top_words.csv")
df = pd.read_csv(path)

newspaper_lists = {col: df[col].tolist() for col in df.columns}

common_words = set(newspaper_lists[df.columns[0]])
for col in df.columns[1:]:
    common_words &= set(newspaper_lists[col])

filtered_lists = {}
for col in df.columns:
    filtered_lists[col] = [word for word in newspaper_lists[col] if word not in common_words]

final_lists = {col: filtered_lists[col][:100] for col in df.columns}

df_new = pd.DataFrame.from_dict(final_lists)
df = df_new

top_10_words = df.iloc[:10, 0].tolist()

heatmap_data = []

for col in df.columns:
    words_of_the_day = df[col].tolist()
    positions = [words_of_the_day.index(word) if word in words_of_the_day else 100 for word in top_10_words]
    heatmap_data.append(positions)

heatmap_data = list(map(list, zip(*heatmap_data)))

df_heatmap = pd.DataFrame(heatmap_data, index=top_10_words, columns=df.columns)

plt.figure(figsize=(20, 10))
sns.heatmap(df_heatmap, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Position in Ranking'}, annot_kws={"size": 8})
plt.title('Position of Top 10 Words from First Day in Subsequent Days')
plt.xlabel('Days')
plt.ylabel('Words')
plt.xticks(rotation=90)
plt.yticks(rotation=0)

output_path = os.path.join(os.getcwd(), "PLOTS", "word_frequency_decay.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')

plt.show()

