import transformers
import pickle
import random

# Define parameters
TRAIN_TEST_SPLIT = 0.8
SEED = 42

# Load the raw data
with open('raw_data.pkl', 'rb') as handle:
    raw_data = pickle.load(handle)

# Tokenize the previews
tokenized_previews = {}
tokenizer = transformers.AutoTokenizer.from_pretrained("ufal/robeczech-base")
no_data, no_preview = [], []
for idx in raw_data:
    if raw_data[idx] is not None:
        preview = raw_data[idx][-1]
        if len(preview): # Consider only previews with non-zero length (preview is available)
            # Clean the preview
            preview = preview.replace('\n', '').replace('\t', '').replace('...', '') # Remove newline characters and dots at the end
            if preview[-6:] == '(více)': # Remove irrelevant text at the end
                preview = preview[:-6]

            # Tokenize the preview
            encoded = tokenizer(preview, truncation='only_first', max_length=512)

            # Get rating
            rating = float(raw_data[idx][6].replace('%', ''))

            # Store the data for the current observation
            tokenized_previews[idx] = (tuple(encoded.input_ids), rating)
        else:
            no_preview.append(idx)
    else:
        no_data.append(idx)

    print(f'Observation {idx+1}/{len(raw_data)} encoded', end='\r')

# Store tokenized previews
with open('tokenized_previews.pkl', 'wb') as out:
    pickle.dump(tokenized_previews, out)

# Print information about preview availability
print(f'Available previews: {len(tokenized_previews)}/{len(raw_data)} ({len(tokenized_previews) / len(raw_data):.2%})')
print(f'Unavailable previews: {len(no_preview)}/{len(raw_data)} ({len(no_preview) / len(raw_data):.2%})')
#print(f'Unavailable data: {len(no_data)}/{len(raw_data)} ({len(no_data) / len(raw_data):.2%})')
print('\n')

# Generate ids for the training and testing sets
ids = {'train':{}, 'test':{}}
train_size = int(0.8 * len(raw_data))
random.seed(SEED)
ids_train = random.sample(range(len(raw_data)), k=train_size)
ids_test = list(set(range(len(raw_data))) - set(ids_train))
ids['train']['with_preview'] = list(set(ids_train) - set(no_preview))
ids['train']['no_preview'] = list(set(no_preview) & set(ids_train))
ids['test']['with_preview'] = list(set(ids_test) - set(no_preview))
ids['test']['no_preview'] = list(set(no_preview) & set(ids_test))
with open('ids_split.pkl', 'wb') as out: # Export the ids
    pickle.dump(ids, out)

# Print information about the preview distribution
print(f'Previews in the train set: {len(ids["train"]["with_preview"])}/{train_size} ({len(ids["train"]["with_preview"])/train_size:.2%})')
print(f'Previews in the test set: {len(ids["test"]["with_preview"])}/{len(ids_test)} ({len(ids["test"]["with_preview"])/len(ids_test):.2%})')