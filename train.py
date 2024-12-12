import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
import torch.nn as nn

import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.optim import AdamW
from data.trust_dataset import *
from torch.utils.data import DataLoader, random_split
from model import *

def main():
    curr_dir = os.getcwd()
    print(curr_dir)
    data_path = os.path.join(curr_dir, "data/clean_newslens_data.csv")
    model_name = "bert-base-uncased"
    num_epochs = 5
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataframe = pd.read_csv(data_path)
    dataset = TrustDataset(dataframe, tokenizer)
    
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    #device = torch.device("cpu")
    print(device)
    
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=True
    )

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    # initialize model
    model = TrustPredictor(model_name, 0.3)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        for _, batch in enumerate(tqdm(train_dataloader)):
        
            tokenized = batch["tokenized"]
            input_ids, attention_mask = tokenized["input_ids"], tokenized["attention_mask"]
            input_ids = input_ids.squeeze(1).to(device)
            print(input_ids.dtype)
            print(attention_mask.dtype)
            print(batch["label"].dtype)
            attention_mask = attention_mask.squeeze(1).to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            output = model({"input_ids": input_ids, "attention_mask": attention_mask}).squeeze(-1)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            


if __name__ == "__main__":
    main()