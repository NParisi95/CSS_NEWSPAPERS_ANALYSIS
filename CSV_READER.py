# =====================================
# FILE PROCESSING
# =====================================

import pandas as pd
import os

def clean_date_format(df):
    df['published_date'] = df['published_date'].astype(str).str[:10]
    return df

def remove_duplicates(df):
    columns_to_ignore = ['Summary', 'keywords']
    columns_to_consider = [col for col in df.columns if col not in columns_to_ignore]
    
    df_cleaned = df.drop_duplicates(subset=columns_to_consider).reset_index(drop=True)
    return df_cleaned

def merge_csv_files(input_folder, output_file):
    dataframes = []
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)
            df = pd.read_csv(file_path)
            dataframes.append(df)

    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df.to_csv(output_file, index=False)

def generate_date_range(df, date_range, output_file):
    start_date, end_date = date_range
    date_list = pd.date_range(start=start_date, end=end_date).strftime('%Y-%m-%d').tolist()
    
    df_filtered = df[df["published_date"].isin(date_list)]
    df_filtered = df_filtered[~df_filtered['testata'].isin(['La Nazione', 'La Gazzetta dello Sport', 'Il Tempo'])]
    df_filtered = df_filtered.dropna(subset=['Text'])
    
    df_filtered.to_csv(output_file, index=False)

def first_page_only(df, output_file):
    to_maintain = {
        "Il Resto del Carlino": "https://www.ilrestodelcarlino.it",
        "L'Unità": "https://www.unita.it",
        "La Stampa": "https://www.lastampa.it",
        "La Repubblica": "https://www.repubblica.it",
        "La Gazzetta dello Sport": "https://esports.gazzetta.it",
        "Il Tempo": "https://www.iltempo.it",
        "Il Sole 24 Ore": "https://www.ilsole24ore.com",
        "Il Giornale": "https://www.ilgiornale.it",
        "Il Foglio": "https://www.ilfoglio.it",
        "Il Fatto Quotidiano": "https://www.ilfattoquotidiano.it",
        "Corriere della Sera": "http://www.corriere.it",
        "Avvenire": "https://www.avvenire.it",
        "Il Post": "https://www.ilpost.it",
        "Ansa": "https://www.ansa.it"
    }
    
    result_df = pd.DataFrame()
    
    for newspaper, url in to_maintain.items():
        subset = df[(df["testata"] == newspaper) & (df["Source"] == url)]
        result_df = pd.concat([result_df, subset], ignore_index=True)
    
    result_df.to_csv(output_file, index=False)

def lemmatize_text(text):
    global nlp
    doc = nlp(text)
    lemmatized_text = ' '.join(token.lemma_ for token in doc)
    return lemmatized_text

def remove_repeated_ngrams(text, repeated_ngrams, min_n, max_n):
    words = tokenize(text)
    text_length = len(words)
    to_remove = set()
    for n in range(max_n, min_n - 1, -1):
        ngrams_list = generate_ngrams(words, n)
        for i, ngram in enumerate(ngrams_list):
            if ngram in repeated_ngrams:
                to_remove.update(range(i, i + n))

    filtered_words = [words[i] for i in range(text_length) if i not in to_remove]
    return ' '.join(filtered_words)




import pandas as pd
import string
import spacy


df = df.dropna(subset=['Text_lemm'])

df["Text_lemm"] = df["Text_lemm"].apply(lower)

nlp = spacy.load("it_core_news_sm")

special_characters = set(string.punctuation)

additional_special_characters = [
    '¡', '¢', '£', '¤', '¥', '¦', '§', '¨', '©', 'ª', '«', '¬', '®', '¯', '°', '±', '²', '³',
    '´', 'µ', '¶', '·', '¸', '¹', 'º', '»', '¼', '½', '¾', '¿', '×', '÷', '€', '‰', '‱', '†',
    '‡', '•', '‣', '․', '‥', '…', '′', '″', '‴', '‵', '‶', '‷', '‹', '›', '‼', '‽', '‾', '⁁',
    '⁂', '⁃', '⁄', '⁅', '⁆', '⁇', '⁈', '⁉', '⁊', '⁋', '⁌', '⁍', '⁎', '⁏', '⁐', '⁑', '⁒', '⁓',
    '⁔', '⁕', '⁖', '⁗', '⁘', '⁙', '⁚', '⁛', '⁜', '⁝', '⁞', '¤', '₠', '₡', '₢', '₣', '₤', '₥',
    '₦', '₧', '₨', '₩', '₪', '₫', '€', '₭', '₮', '₯', '₰', '₱', '₲', '₳', '₴', '₵', '₶', '₷', '₸',
    '₹', '₺', '₻', '₼', '₽', '₾', '℀', '℁', 'ℂ', '℃', '℄', '℅', '℆', 'ℇ', '℈', '℉', 'ℊ', 'ℋ',
    'ℌ', 'ℍ', 'ℎ', 'ℏ', 'ℐ', 'ℑ', 'ℒ', 'ℓ', '℔', 'ℕ', '№', '℗', '℘', 'ℙ', 'ℚ', 'ℛ', 'ℜ', 'ℝ',
    '℞', '℟', '℠', '℡', '™', '℣', 'ℤ', '℥', 'Ω', '℧', 'ℨ', '℩', 'K', 'Å', 'ℬ', 'ℭ', '℮', 'ℯ',
    'ℰ', 'ℱ', 'Ⅎ', 'ℳ', 'ℴ', 'ℵ', 'ℶ', 'ℷ', 'ℸ', 'ℹ', '℺', '℻', 'ℼ', 'ℽ', 'ℾ', 'ℿ', '⅀', '⅁',
    '⅂', '⅃', '⅄', 'ⅅ', 'ⅆ', 'ⅇ', 'ⅈ', 'ⅉ', '⅊', '⅋', '⅌', '⅍', 'ⅎ', '⅏', '⅐', '⅑', '⅒', '⅓',
    '⅔', '⅕', '⅖', '⅗', '⅘', '⅙', '⅚', '⅛', '⅜', '⅝', '⅞', '⅟', 'Ⅰ', 'Ⅱ', 'Ⅲ', 'Ⅳ', 'Ⅴ', 'Ⅵ',
    'Ⅶ', 'Ⅷ', 'Ⅸ', 'Ⅹ', 'Ⅺ', 'Ⅻ', 'Ⅼ', 'Ⅽ', 'Ⅾ', 'Ⅿ', 'ⅰ', 'ⅱ', 'ⅲ', 'ⅳ', 'ⅴ', 'ⅵ', 'ⅶ',
    'ⅷ', 'ⅸ', 'ⅹ', 'ⅺ', 'ⅻ', 'ⅼ', 'ⅽ', 'ⅾ', 'ⅿ', 'ↀ', 'ↁ', 'ↂ', 'Ↄ', 'ↄ', 'ↅ', 'ↆ', 'ↇ', 'ↈ'
]

special_characters.update(additional_special_characters)

def remove_verbs_from_text(text):
    doc = nlp(text)
    words_without_verbs = []

    stopwords = {"another", "to do", 'there', 'fa', 'so much', 'however', 'here it is', 'always', 'because', 'goes', 'that', 'boh', 'among',
                        'del', 'della', 'dello', 'dell', 'degli', 'delle', 'dei',
                        'to the', 'to the', 'to the', 'to the', 'to the', 'to the', 'to the',
                        'from the', 'from the', 'from the', 'from the', 'from the', 'from the', 'from the',
                        'with the', 'with the', 'with the', 'with the',
                        'on the', 'on the', 'on the', 'on the', 'on the', 'on the', 'on the',
                        'after', 'https', 'then', 'to see', 'you', 'quest', 'give', 'no', 'but more', 'when', 'state',
                        'now', 'every', 'so', 'be', 'fact', 'be', 'today', 'can', 'touch', 'want',
                        'other', 'therefore', 'great', 'alone', 'now', 'thanks', 'thing', 'already', 'me', '-', 'can',
                        'other', 'before', 'year', 'pure', 'here', 'will be', 'own', 'knows', 'of', 'silvio', 'roberto', 'renzi', 'calenda', 'good morning', 'good evening',
                        'new', 'much', 'puts', 'say', 'such', 'can', 'use', 'that is', 'high', 'do', 'any', 'so much', 'so much', 'tonight', 'evening',
                        'so', 'call', 'understood', 'never', 'have', 'go', 'instead', 'months', 'still','rtl', 'especially', 'above all', 'greetings',
                        'instead', 'talk', 'go','allegri', 'qsta', 'qsto', 'anch', 'prch', 'com', 'snza', 'said', 'qlli', 'no', 'said','says',
                        'someone','any','which', 'yesterday','today', 'ile','cio','another','away', 'appero', 'hours', 'facebook', 'evening',
                        'will be', 'interviewed', 'live', 'rae', 'zonabianca', 'mezzorainpiu', 'portaaporta', 'friends',
                        'almost','will go','https','must','will have','nun','not', 'accounthttps','etc', 'wants','sti','here','nor','beyond','wants','I wonder',
                        'this', 'this', 'followed', 'follow me', 'tomorrow', 'sunday vote league', 'tomorrow vote league', #'political elections', 'votafdi',
                        'square','rome','turin','milan', 'naples', 'ancona','catania', 'sassari', 'florence', 'palermo', 'perugia', 'sassari', 'genoa',
                        'August', 'September','nothing','well','Saturday','Sunday', 'few','years','many', 'two','three', 'five', '25', 'vote', 'vote'}

    for token in doc:
        if token.text.lower() not in stopwords and not token.is_punct and not token.like_num and token.is_alpha and token not in special_characters:
            words_without_verbs.append(token.text)
    
    return " ".join(words_without_verbs)

df["Text_lemm"] = df["Text_lemm"].apply(remove_verbs_from_text)


def main():
    input_folder = os.path.join(os.getcwd(), "ARTICLES DATA", "COMPLETE NEWSPAPER")
    output_merged_file = os.path.join(os.getcwd(), "ARTICLES DATA", "MERGED DATASET", "merged_output_RAW.csv")
    output_clean_file = os.path.join(os.getcwd(), "ARTICLES DATA", "MERGED DATASET", "merged_output_CLEAN.csv")
    output_first_page_file = os.path.join(os.getcwd(), "ARTICLES DATA", "MERGED DATASET", "merged_output_FIRST_PAGE_ONLY.csv")
    
    span_value = ["2024-06-06", "2024-07-15"]
    
    print("\n STEP 1")
    merge_csv_files(input_folder, output_merged_file)
    
    print("\n STEP 2")
    df_merged = pd.read_csv(output_merged_file)
    df_cleaned = clean_date_format(df_merged)
    df_cleaned = remove_duplicates(df_cleaned)
    generate_date_range(df_cleaned, span_value, output_clean_file)
    
    print("\n STEP 3")
    df_cleaned = pd.read_csv(output_clean_file)
    first_page_only(df_cleaned, output_first_page_file)
    
    print("\n STEP 4")
    df_fp_only = pd.read_csv(output_first_page_file)
    nlp = spacy.load('it_core_news_sm')
    
    df_fp_only['Text_lemm'] = df_fp_only['Text'].apply(lemmatize_text)
    df_fp_only['Title_lemm'] = df_fp_only['Title'].apply(lemmatize_text)
    
    df_fp_only.to_csv(output_first_page_file, index=False)
    
    print("\n STEP 5")
    df_final = pd.read_csv(output_first_page_file)
    nltk.download('punkt')
    

if __name__ == "__main__":
    main()
