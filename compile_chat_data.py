#!/bin/python
# compile_chat_data.py - update this
# usage: - update this
# Mark Lekina Rorat, Thu Sep 23 16:17:32 EDT 2021

# import libraries
import os
import random
import re
import pandas as pd
from build_dataset import clean_text, write_to_file

# declare global variables
metadata_regex = "\[\d+\/\d+\/\d+,\s\d+:\d+:\d+\s\w+\][\s\w+]*:"
grouped_metadata_regex = "\[(\d+\/\d+\/\d+),\s(\d+:\d+:\d+\s\w+)\]([\s\w+]*):"

# set random seed
random.seed(0)


# function to extract and compile chat data from chat text files
def process_chat_data(source_dir, dest_file):
    # generate paths for all chat files in source_dir
    paths = []
    for filename in os.listdir(source_dir):
        if filename.endswith(".txt"):
            path = os.path.join(source_dir, filename)
            paths.append(path)

    # loop through each path, extract and compile data to clean_data
    clean_data = []
    for path in paths:
        with open(path, 'r') as input:
            chat = input.read()

        # extract regex matches
        # extract text lines and drop first blank line
        matches = re.findall(grouped_metadata_regex, chat)
        lines = re.split(metadata_regex, chat)
        lines.pop(0)

        # extract metadata from each match
        # match metadata with corresponding text line
        # append the result to clean_data list
        for index, match in enumerate(matches):
            date, time, author = match
            date, time, author = date.strip(), time.strip(), author.strip()
            line = clean_text(lines[index])
            clean_data.append([date, time, author, line])

    # shuffle the data and generate a dataframe
    # write dataframe to file
    random.shuffle(clean_data)
    df = pd.DataFrame(clean_data, columns=['date', 'time', 'author', 'line'])
    write_to_file(df, dest_file)


if __name__ == "__main__":
    process_chat_data('data/whatsapp_chat_data', 'data/chat_data.csv')
