import pickle
import pandas as pd
import re
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning) # Ignore warning for inserting too many  columns

# Specify parameters
NO_OF_FEATURES = 13

# Load the raw data
with open('raw_data.pkl', 'rb') as handle:
    raw_data = pickle.load(handle)

# Load train and test ids
with open('ids_split.pkl', 'rb') as handle:
    ids = pickle.load(handle)
for val in ['train', 'test']:
    for val2 in ['with_preview', 'no_preview']:
        ids[val][val2] = set(ids[val][val2])

# Load preview model predictions
with open('preview_preds.pkl', 'rb') as handle:
        preview_preds = pickle.load(handle)

# Process the basic data
processed_data = []
for idx in raw_data:

    # Prepare a list of results for the current observation
    obs_data = [None] * (NO_OF_FEATURES - 2) # Skip title and preview

    # Extract the genres
    obs_data[0] = tuple(raw_data[idx][1].split(' / '))

    # Extract the country
    obs_data[1] = tuple(raw_data[idx][2].split(' / '))
    if obs_data[1][0] == '':
        obs_data[1] = tuple(['Neznámo'])

    # Extract the year
    obs_data[2] = int(raw_data[idx][3])

    # Extract the duration - NEED EXCEPTION FOR one observation where country is part of duration
    duration = re.search('^([0-9]*)', raw_data[idx][4]) # Find duration
    if duration is not None and len(duration.groups()[0]):
        obs_data[3] = int(duration.groups()[0])
    else:
        obs_data[3] = int(re.search(' ([0-9]*) ', raw_data[idx][4]).groups()[0])

    # Extract the director
    obs_data[4] = raw_data[idx][5]

    # Extract the number of ratings
    obs_data[5] = int(raw_data[idx][7][1:-1].replace(' ', ''))

    # Extract the number of fans
    obs_data[6] = int(raw_data[idx][8][1:-1].replace(' ', ''))

    # Extract the ratings
    obs_data[7] = float(raw_data[idx][6].replace('%', ''))

    # Add indicator for training
    obs_data[8] = 1 if (idx in (ids['train']['no_preview'] | ids['train']['with_preview'])) else 0

    # Add an indicator if preview is available
    obs_data[9] = int(len(raw_data[idx][-1]) > 0)

    # Add a prediction of the preview model
    obs_data[10] = preview_preds[idx]

    # Store the observation
    processed_data.append(obs_data) 

    # Print info
    print(f'Observation {idx+1}/{len(raw_data)} processed', end='\r')

# Create a pandas data frame for the data
processed_data = pd.DataFrame(processed_data, columns=['Genres', 'Country', 'Year', 'Duration', 'Director', 'Number of ratings', 'Number of fans', 'Ratings', 'Train', 'Preview', 'Preview score'])
processed_data.to_csv('raw_data.csv', index=False, sep=';') # Store the raw data set

# Process genres and countries
for col in ['Genres', 'Country']:
    processed_data[col] = processed_data[col].apply(set)
    uni_vals = set()
    for idx in range(processed_data.shape[0]): # Find all unique values
        uni_vals = uni_vals | processed_data.loc[idx, col]
    processed_data[list(uni_vals)] = 0 # Make a column for each unique value
    for idx in range(processed_data.shape[0]): # Fill each unique column
        for uni_val in processed_data.loc[idx, col]:
            processed_data.loc[idx, uni_val] = 1
    processed_data.drop(columns=col, inplace=True) # Drop the raw column

# Drop one column to avoid perfect collinearity
processed_data.drop(columns='Angola', inplace=True)

# Add the number of movies directed so far for each director
processed_data.sort_values(['Director', 'Year'], inplace=True)
processed_data['Number of directed movies'] = processed_data.groupby('Director').cumcount()
processed_data['Number of directed movies'] = processed_data['Number of directed movies'].fillna(0)
processed_data.drop(columns='Director', inplace=True)

# Process year
processed_data['Age'] = 2023 - processed_data['Year'] # Calculate age
processed_data.drop(columns='Year', inplace=True)

# Assign mean ratings to observations with unavailable previews
mean_no_preview = processed_data.loc[processed_data['Train'] & (processed_data['Preview'] == 0), 'Ratings'].mean()
processed_data['Preview score'] = processed_data['Preview score'].fillna(mean_no_preview)

# Export the data
processed_data.to_csv('processed_data.csv', index=False, sep=';')

    





