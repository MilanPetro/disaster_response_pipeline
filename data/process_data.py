import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Loading the messages and categories files and merging them together.
    Input: filepath to messages, filepath to categories
    Output: merged dataframe of messages and categories data"""

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")
    return df

def clean_data(df):
    """Cleaning the message and categories data.
    Input: merged dataframe of messages and categories data"""

    # Splitting categories into separate category columns:
    # creating a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(";", expand=True)

    # selecting the first row of the categories dataframe to use this row
    # to extract a list of new column names for categories.
    row = categories.iloc[0,:]
    category_colnames = row.str[:-2]
    categories.columns = category_colnames

    # converting category values to numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = pd.to_numeric(categories[column])

    # dropping the original categories column from `df`
    df.drop("categories", axis=1, inplace=True)

    # concatenating the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)

    # dropping rows with non 0/1 from the dataframe and dropping duplicates
    df.drop(df[df[(df.iloc[:,4:]!=0)&(df.iloc[:,4:]!=1)].any(axis=1)].index,
            inplace=True)
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """Saving data to sqlite database
    Input: cleaned dataframe, filename for database"""

    engine = create_engine('sqlite:///{}'.format(str(database_filename)))
    df.to_sql("messages", engine, index=False, if_exists="replace")

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()
