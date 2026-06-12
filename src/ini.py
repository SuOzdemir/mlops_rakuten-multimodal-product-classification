import pandas as pd
import numpy as np
from pathlib import Path
import re
import nltk
# nltk.download('stopwords')  # this could be neccessary
from nltk.corpus import stopwords

stop_words = set(stopwords.words(['french', 'english', 'german']))

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR  = DATA_DIR / "raw"

train_dir = RAW_DIR / "images" / "image_train"
test_dir = RAW_DIR / "images" / "image_test"

def get_path(row):
    fname = f"image_{row.imageid}_product_{row.productid}.jpg"
    
    p1 = train_dir / fname
    if p1.exists():
        return str(p1)
    
    p2 = test_dir / fname
    if p2.exists():
        return str(p2)
    
    return None

# removing repeating text blocks in description

def remove_repeated_blocks(text, min_block_len=100):
    """
    Remove consecutively repeated text blocks of length >= min_block_len.
    Keeps the first occurrence only.
    """
    if not isinstance(text, str):
        return text

    text = text.strip()
    if len(text) < 2 * min_block_len:
        return text

    # build regex safely
    pattern = re.compile(r"(.{" + str(min_block_len) + r",}?)(?:\1)+", re.DOTALL)

    previous = None
    while text != previous:
        previous = text

        # collapse repeated consecutive blocks
        text = pattern.sub(r"\1", text)

        # normalize spaces created during replacement
        text = re.sub(r"\s+", " ", text).strip()

    return text

def clean_txt_colmn(dtfrme, column):    # function to clean columns in a dataframe and generate a new column_clean 
    new = column + "_clean"    
    # replacing NaNs with empty strings
    dtfrme[new] = dtfrme[column].fillna("")    
    # remove html tags
    dtfrme[new] = dtfrme[new].str.replace(r"<.*?>", " ", regex=True)    
    # decode html entities &amp
    dtfrme[new] = dtfrme[new].str.replace(r"&\w+;", " ", regex=True)    
    # lowercase
    dtfrme[new] = dtfrme[new].str.lower()    
    # remove punctuation and special characters (parantheses ...)
    dtfrme[new] = dtfrme[new].str.replace(r"[^\w\s]", " ", regex=True)        
    # normalize spaces
    dtfrme[new] = dtfrme[new].str.replace(r"\s+", " ", regex=True) 
    # strip
    dtfrme[new] = dtfrme[new].str.strip()
    return dtfrme

def prepare_all_words(df, column, stop_words): # function to prepare vocabularies
    # tokenize
    tokens = df[column].str.split()
    # flatten into single column of words
    words = tokens.explode()
    # remove stopwords
    words = words[~words.isin(stop_words)]
    # remove short tokens
    words = words[words.str.len() > 2]
    return words

def remove_numeric_tokens(text):
    if not isinstance(text, str):
        return text
    tokens = text.split()
    tokens = [t for t in tokens if not t.isdigit()]
    return " ".join(tokens)

X_train = pd.read_csv(RAW_DIR / "X_train.csv")
Y_train = pd.read_csv(RAW_DIR / "Y_train.csv")
X_test = pd.read_csv(RAW_DIR / "X_test.csv")

df = pd.merge(X_train, Y_train, on="Unnamed: 0")

df["image_path"] = df.apply(get_path, axis=1)

clean_txt_colmn(df,"designation")

clean_txt_colmn(df,"description")

# remove repeated block of texts only to descriptions
df["description_dedup"] = df["description_clean"].apply(remove_repeated_blocks)

# create new, cleaned columns without digits
df["designation_nodigits"] = df["designation_clean"].apply(remove_numeric_tokens)
df["description_nodigits"] = df["description_dedup"].apply(remove_numeric_tokens)

df["text_combined"] = (df["designation_clean"] + " " + df["description_dedup"].fillna("")).str.strip()

df.to_csv(RAW_DIR / "train_clean.csv", index=False)

