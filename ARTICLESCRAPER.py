# =====================
# ARTICLES SCRAPER
# =====================

import json
import newspaper
from newspaper import Article
from newspaper import Source
import pandas as pd
import os
import time
from datetime import datetime

def read_sources(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    sources_dict = {source['name']: source['url'] for source in data['sources']}
    return sources_dict

# Example usage
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'sources.json')
sources = read_sources(file_path)

data = []
start_time = time.time()

for name, url in sources.items():
    source = newspaper.build(url, memoize_articles=False) 
    print("found in", name, len(source.articles))
    
    failed_download = 0
    for article in source.articles:
        try:
            article.download()
            article.parse()
            article.nlp()
            
            temp_dict = {
                'source': name,
                'Title': article.title,
                'Authors': ', '.join(article.authors),
                'Text': article.text,
                'Summary': article.summary,
                'published_date': article.publish_date,
                'Source': article.source_url,
                'keywords': ', '.join(article.keywords),
                'category': None
            }

            for category_url in source.category_urls():
                if temp_dict["category"] is None:
                    temp_dict['category'] = str(category_url)
                else:
                    temp_dict["category"] += str(category_url)
                    
            data.append(temp_dict)

        except Exception as e:
            failed_download += 1
            continue

    print("failed downloads for", name, failed_download)  

def today_date():
    today = datetime.now()
    formatted_date = today.strftime("%d-%m-%Y")
    return formatted_date

df = pd.DataFrame(data)
output_file = os.path.join(current_dir, f'articles - {today_date()}.csv')
df.to_csv(output_file, index=False, encoding='utf-8')

print(f"CSV saved as {output_file}")

end_time = time.time()
execution_time = (end_time - start_time) / 60
print(f"Total execution time: {execution_time:.2f} minutes")
