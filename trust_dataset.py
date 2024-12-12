import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

class TrustDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        super(TrustDataset, self).__init__()
        self.tokenizer = tokenizer
        self.df = dataframe

    def __getitem__(self, idx):
        label = self.df.rating_scale_response.iloc[idx].astype(np.float32)
        content = self.df.extracted_content_body.iloc[idx]
        q26 = self.df.q26_trust_media.iloc[idx]
        content_title = self.df.content_title_clean.iloc[idx]
        channel_id = self.df.channel_id.iloc[idx]
        user_id = self.df.external_user_id.iloc[idx]
        tokenized = self.tokenizer(
            content,
            return_tensors="pt",
            truncation=True,
            padding="max_length"
        )

        return {
            "label": label,
            "content": content,
            "q26": q26,
            "content_title": content_title,
            "channel_id": channel_id,
            "user_id": user_id,
            "tokenized": tokenized
        }

    def __len__(self):
        return len(self.df)