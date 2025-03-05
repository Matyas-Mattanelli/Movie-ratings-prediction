from trainable_module import TrainableModule
import torch
import torchmetrics
import pickle
import numpy as np
import transformers

# Define the parameters
HIDDEN_SIZE = 64
BATCH_SIZE = 5
EPOCHS = 5
SEED = 42
TRAIN = True

# Define the data set class
class TokenizedPreviews(torch.utils.data.Dataset):
    
    def __init__(self, kind):
        # Specify the kind
        self.kind = kind

        # Load all the data
        with open('tokenized_previews.pkl', 'rb') as handle:
            df = pickle.load(handle)

        # Load train and test ids
        with open('ids_split.pkl', 'rb') as handle:
            ids = pickle.load(handle)

        # Select only required subset
        ids = set(ids[kind]['with_preview']) # Get the ids for the appropriate data set and turn it into a set
        self._size = len(ids) # Get the size of the data set
        self.df = [df[idx] for idx in df if idx in ids] # Extract only relevant observations

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int):
        return self.df[index]

# Define the model class
class Model(TrainableModule):
    def __init__(self, robeczech):
        super().__init__()

        # Define the model
        self.robeczech = robeczech
        self.hidden_layer = torch.nn.Linear(robeczech.config.hidden_size, HIDDEN_SIZE)
        self.relu = torch.nn.ReLU()
        self.output_layer = torch.nn.Linear(HIDDEN_SIZE, 1)

        # Initialize weights
        torch.nn.init.xavier_uniform_(self.hidden_layer.weight)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)

    # Implement the model computation.
    def forward(self, token_ids, attention_mask) -> torch.Tensor:
        inputs = self.robeczech(token_ids, attention_mask=attention_mask)
        hidden = self.hidden_layer(inputs[1])
        hidden = self.relu(hidden)

        return self.output_layer(hidden)

if __name__ == '__main__':

    # Load the robeczech model
    robeczech = transformers.AutoModel.from_pretrained("ufal/robeczech-base")

    # Create datasets
    def prepare_batch(batch):
        token_ids, ratings = zip(*batch)
        token_ids = [torch.tensor(obs) for obs in token_ids]
        token_ids = torch.nn.utils.rnn.pad_sequence(token_ids, batch_first=True, padding_value=1)
        attention_mask = token_ids != 1

        return (token_ids, attention_mask), torch.tensor(ratings)[:, None]

    test = torch.utils.data.DataLoader(TokenizedPreviews('test'), BATCH_SIZE, shuffle=False, collate_fn=prepare_batch)

    # Create the model
    model = Model(robeczech)

    if TRAIN:

        # Set the random seed
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        # Create the training data set
        train = torch.utils.data.DataLoader(TokenizedPreviews('train'), BATCH_SIZE, shuffle=True, collate_fn=prepare_batch)

        # Freeze the pretrained model
        for param in model.robeczech.parameters():
            param.requires_grad = False

        # Train the newly added layers
        model.configure(
            optimizer=torch.optim.Adam(model.parameters()),
            loss=torch.nn.MSELoss(),
            metrics=torchmetrics.regression.MeanAbsoluteError()
        )
        model.fit(train, dev=test, epochs=EPOCHS)

        # Unfreeze the pretrained model for finetuning
        for param in model.robeczech.parameters():
            param.requires_grad = True

        # Fine tune
        model.configure(
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-5),
            loss=torch.nn.MSELoss(),
            metrics=torchmetrics.regression.MeanAbsoluteError()
        )
        model.fit(train, dev=test, epochs=EPOCHS)

        # Save the model
        model.save_weights('preview_model.pt')

    else:
        # Load already trained model weights
        model.load_weights('preview_model.pt')
        model = model.cuda()

    # Make predictions
    train_no_shuffle = torch.utils.data.DataLoader(TokenizedPreviews('train'), BATCH_SIZE, shuffle=False, collate_fn=prepare_batch)
    train_preds = model.predict(train_no_shuffle)
    test_preds = model.predict(test)

    # Load split ids
    with open('ids_split.pkl', 'rb') as handle:
            ids = pickle.load(handle)
    preview_preds = {}

    # Store training predictions (observations with preview)
    for idx_pred, idx_main in enumerate(ids['train']['with_preview']):
        preview_preds[idx_main] = train_preds[idx_pred][0]

    # Store testing predictions (observations without preview)
    for idx_pred, idx_main in enumerate(ids['test']['with_preview']):
        preview_preds[idx_main] = test_preds[idx_pred][0]

    # Specify that there is no prediction for predictions without preview
    for idx_main in ids['train']['no_preview'] + ids['test']['no_preview']:
        preview_preds[idx_main] = None

    # Store predictions
    with open('preview_preds.pkl', 'wb') as out:
        pickle.dump(preview_preds, out)
    