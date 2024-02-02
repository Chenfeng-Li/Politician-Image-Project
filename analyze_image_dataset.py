import numpy as np
import pandas as pd
import sys

from tqdm import tqdm


def split_url(url):
    """ Split url by . and / """
    parts = url.split('/')
    split_parts = [item for part in parts for item in part.split('.')]
    return split_parts

def valid_url(url):
    """ 
    Determine if a url is valid.
    A URL starts with "http:" or "https:" is considered as valid 
    """
    split = split_url(url)
    return "http:" in split or "https:" in split


def holder(split):
    """ 
    Find the holder (abbrevation) of a URL.
    For a url split, the string before "com", "org" or "net" is considered as the holder.
    Omit other top-level domains. Those are minority.
    """

    if "com" in split:
        idx = split.index("com")
    elif "org" in split:
        idx = split.index("org")
    elif "net" in split:
        idx = split.index("net")
    else:
        cor_list.append('Not Specify')
        return
    cor.add(split[idx-1])
    cor_list.append(split[idx-1])


# According to the abbrevations, we find the following map of corporation
news_corporations = {
    "afp": "Agence France-Presse",
    "apnews": "Associated Press News",
    "breitbart": "Breitbart News",
    "cnn": "Cable News Network",
    "dailycaller": "The Daily Caller",
    "foxbusiness": "Fox Business Network",
    "foxnews": "Fox News Channel",
    "jns": "Jewish News Syndicate",
    "jpost": "The Jerusalem Post",
    "mediaite": "Mediaite",
    "npr": "National Public Radio",
    "politico": "Politico",
    "politicopro": "PoliticoPro",
    "thehill": "The Hill",
    "timesofisrael": "The Times of Israel",
    "washingtontimes": "The Washington Times",
    "washtimes": "The Washington Times",
    "wp": "The Washington Post",
    "wsj": "The Wall Street Journal"
}

def valid_cor(abbr):
    """ Determine if a abbrevation is from valid corporation. """
    return abbr in news_corporations.keys()



if __name__ == "__main__":
    print("Loading Dataset images-20230907.csv.")
    try:
        df = pd.read_csv("images-20230907.csv",header=None)
    except FileNotFoundError:
        print("Dataset images-20230907.csv not found. Download first.")
        sys.exit(1)   

    # Rename Columns
    df.columns= ["UUID", "Title", "URL", "Caption", "Text", "ImageType"]

    # Drop rows with no URL
    df = df.dropna(subset=["URL"])

    # Drop rows with invalid URL
    df = df[df["URL"].apply(valid_url)]

    # Drop rows with duplicate URL
    df = df.drop_duplicates(subset=["Title", "URL", "Caption", "Text"])

    print(f"Shape of Dataset: {df.shape}\n")


    # Determine the corporation of each image
    # Add a new column "Corporation"
    cor = set()
    cor_list = []
    for idx, url in tqdm(enumerate(df["URL"])):
        split = split_url(url)
        holder(split)
    df['Corporation'] = cor_list

    print(f"Corporations from the dataset:\n{cor}\n")
    print(f"Valid Corporation:")
    for key, val in news_corporations.items():
        print(f"{key}: {val}")

    print("Filter rows with valid corporations")
    df = df[df['Corporation'].apply(valid_cor)]
    print(f"Shape of Dataset: {df.shape}\n")
    
    print("Number of samples from each valid corporation")
    print(df['Corporation'].value_counts())

    # Save the dataset
    df.to_csv('image_dataset.csv', index=False)



    




