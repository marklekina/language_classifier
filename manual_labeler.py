#!/bin/python
#
# manual_labeler.py - read unlabelled data from a source file;
#                     display text data line-by-line and prompt user to provide appropriate label;
#                     write labelled data to a new file.
#
# Mark Lekina Rorat, July, Sept, Oct 2021


import random
import pandas as pd
from build_dataset import write_to_file


# input: source filename
# output: list of lists
# extracts unlabelled data from a .csv file and returns a list-of-lists representation of the data

# read lines from file
# filter lines by categorical threshold
# extract text from each line into a list of lists


def extract_unlabelled_data(source_filename, threshold=.05):
    # read rows from file into dataframe
    df = pd.read_csv(source_filename)
    # get labels
    labels = df.predicted_label.unique().tolist()
    # list to hold all extracted data in lists
    list_of_lists = []
    # filter text by labels
    for label in labels:
        # sub-list and dataframe to hold data with common label
        sub_list_of_lists = []
        sub_df = df.loc[df.predicted_label == label]
        # extract data to lists and add lists to sub-list
        for index, row in sub_df.iterrows():
            group = [row.date, row.time, row.author, row.line]
            sub_list_of_lists.append(group)
        # shuffle sub-list and standardize sub-list
        random.shuffle(sub_list_of_lists)
        limit = int(threshold * len(sub_list_of_lists))
        groups = sub_list_of_lists[:limit]
        [list_of_lists.append(group) for group in groups]
    # shuffle combined list
    random.shuffle(list_of_lists)
    return list_of_lists


# input: list of lists
# output: labelled list of lists
# extracts text from a data list, prompts the user to label the text, and returns the labelled data

# print out line and prompt user to enter label (1, 2, 3) for sheng, swahili or english
# read input, match appropriate label and add to list
# add list to labelled list of lists
# return labelled list of lists
def label_text(list_of_lists):
    labelled_list_of_lists = []
    print("Please match the following text with the most appropriate numerical label: (1 -> sheng; 2 -> swahili; 3 -> english)\n")
    for group in list_of_lists:
        value = True
        while value:
            try:
                numerical_label = int(input(group[3] + '\t:\t').strip())
                value = False
            except ValueError:
                print('Please enter a label!')

        if numerical_label == 1:
            labelled_group = group + ['sheng']
        elif numerical_label == 2:
            labelled_group = group + ['swahili']
        elif numerical_label == 3:
            labelled_group = group + ['english']
        else:
            print('Error: no matching label!')
            labelled_group = group + []
        labelled_list_of_lists.append(labelled_group)
    return labelled_list_of_lists


# load dataset, extract and label data, and write it to a .csv file
def main():
    unlabelled_data = extract_unlabelled_data('data/labelled_chat_data.csv')
    labelled_data = label_text(unlabelled_data)
    df = pd.DataFrame(labelled_data, columns=['date', 'time', 'author', 'line', 'gold_label'])
    write_to_file(df, 'data/manually_labelled_chat_data.csv')


if __name__ == "__main__":
    main()
