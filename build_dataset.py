#!/bin/python
#
# build_dataset.py - read data from three language sources (sheng, swahili and english);
#                    balance and merge the data to create a combined dataset;
#                    write dataset to .csv file.
#
# Mark Lekina Rorat, July, Sept, Oct 2021


import os
import random
import re
import pandas as pd

# declare global variables
regex = "[^A-Za-z _\.,!?\"'/]"
sheng_articles = 195
swahili_articles = 116
english_articles = 117

# set random seed
random.seed(0)


# standardize generic text
def clean_text(text):
    text = re.sub(regex, ' ', text)  # substitute non-alphabetical characters with spaces
    text = " ".join(text.split())  # remove extra spaces
    text = text.strip().lower()  # remove newlines from the start and end of the file
    return text


# standardize sheng text
def clean_sheng_text(text):
    sheng_regex = r'(([A-Z]+ +)|[A-Z][A-Z]+)|(People Daily)|(([0-9]+/[0-9]+/[0-9]+).+)|(http.+)|(Page .+)|(MANU ' \
                  r'?SCRIBES|Manu ?scribes)|(URADI|Uradi)|(TAMBU ?LIKA|TAMU ?LIKA|Tamb ?ulika|Tamu ?lika)|(Fomu ?Ni(' \
                  r'Safi)?\??|FOMU ?NI(SAFI)?\??)|(MANUEL NTOYAI|Manuel Ntoyai)|(NAFSI ?HURU|Nafsi ?Huru)|(([A-Z]+ +)|[' \
                  r'A-Z][A-Z]+)|(\n)'
    return re.sub(sheng_regex, ' ', text)


# TODO: standardize this function
# preprocess sheng data
# read text from each file and clean it
# return a list of text-label pairs
def process_sheng_sources(source_dir):
    sheng_pairs = []
    # loop through all files in source_dir
    for filename in os.listdir(source_dir):
        path = os.path.join(source_dir, filename)
        with open(path, 'r') as file:
            text = file.read()
            text = clean_sheng_text(text)
            text = clean_text(text)
            sheng_pairs.append([text, 'sheng'])
    return sheng_pairs


# TODO: test generic function

def process_csv_sources(source_path, column, label):
    pairs = []
    # read source files
    df = pd.read_csv(source_path, usecols=column)
    for index, row in df.iterrows():
        row = clean_text(row[column])
        pairs.append([row, label])
    return pairs


# function to preprocess english data
# read text from source file and clean it
# return a list of text-label pairs
def process_english_sources(source_path):
    english_pairs = []
    # read source file
    df = pd.read_csv(source_path, usecols=["Text"])
    for index, row in df.iterrows():
        text = clean_text(row["Text"])
        english_pairs.append([text, 'english'])
    return english_pairs


# function to preprocess swahili data
# read text from source file and clean it
# return a list of text-label pairs
def process_swahili_sources(source_path):
    swahili_pairs = []
    df = pd.read_csv(source_path, usecols=["content"])
    for index, row in df.iterrows():
        text = clean_text(row["content"])
        swahili_pairs.append([text, 'swahili'])
    return swahili_pairs


# shuffle and standardize list size
def balance_data(a_list, threshold):
    random.shuffle(a_list)
    return a_list[:threshold]


# write dataframe to .csv file
def write_to_file(df, dest_path):
    # drop rows with blank values
    df.replace("", float('NaN'), inplace=True)
    df.dropna(inplace=True)
    # write to file
    df.to_csv(dest_path, encoding='utf-8', index=False)


def main():
    # generate text-label pairs for text in each language
    sheng_pairs = process_sheng_sources('data/sheng_data')
    swahili_pairs = process_swahili_sources('data/swahili_data/swahili_news.csv')
    english_pairs = process_english_sources('data/english_data/bbc_news.csv')

    # balance and merge pair lists
    sheng_pairs = balance_data(sheng_pairs, sheng_articles)
    swahili_pairs = balance_data(swahili_pairs, swahili_articles)
    english_pairs = balance_data(english_pairs, english_articles)
    merged_pairs = sheng_pairs + swahili_pairs + english_pairs

    # shuffle merged pair list and convert to dataframe
    random.shuffle(merged_pairs)
    df = pd.DataFrame(merged_pairs, columns=['text', 'label'])

    # write dataframe to file
    write_to_file(df, 'data/language_dataset.csv')


if __name__ == "__main__":
    main()
