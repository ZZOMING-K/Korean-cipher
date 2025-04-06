import random 
import re 
import torch 
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

def set_seed(seed: int = 42):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def calculate_accuracy(logits, labels, ignore_index=0):

    predicted = torch.argmax(logits, dim=1)
    mask = (labels != ignore_index)  # Ignore padding tokens
    correct = (predicted == labels).masked_select(mask).sum().item()
    total = mask.sum().item()
    
    accuracy = correct / total
    
    return accuracy

def evaluate(model, valid_dataloader, criterion, device, output_size):
    
    val_loss, val_correct, val_total = 0.0, 0.0, 0

    model.eval()
    
    with torch.no_grad():
        
        for input_ids, output_ids in valid_dataloader:
            
            # Move data to device
            input_ids, output_ids = input_ids.to(device), output_ids.to(device)
            
            # Forward pass
            logits = model(input_ids)
            loss = criterion(logits.view(-1, output_size), output_ids.view(-1))
            
            # Update metrics
            val_loss += loss.item()
            batch_accuracy = calculate_accuracy(
                logits.view(-1, output_size), 
                output_ids.view(-1)
            )
            val_correct += batch_accuracy * output_ids.size(0)
            val_total += output_ids.size(0)
            
    val_accuracy = val_correct / val_total
    val_loss /= len(valid_dataloader)
    
    return val_loss, val_accuracy


class TextProcessor:

    def split_sentences(text):

        sentences = text.split('. ')
        sentences = [s + '.' for s in sentences[:-1]] + [sentences[-1]]
        
        return sentences

    def remove_extra_spaces(text):

        return re.sub(r'\s+', ' ', text).strip()

    def preprocess_dataframe(file_path):

        df = pd.read_csv(file_path)

        df = df.drop_duplicates().reset_index(drop=True)

        df['input'] = df['input'].str.strip()
        df['output'] = df['output'].str.strip()


        df['input'] = df['input'].apply(TextProcessor.split_sentences)
        df['output'] = df['output'].apply(TextProcessor.split_sentences)
        df = df.explode(column=['input', 'output']).reset_index(drop=True)
        

        df = df.drop_duplicates().reset_index(drop=True)
        

        df = df[df['input'] != df['output']]

        df['input'] = df['input'].apply(TextProcessor.remove_extra_spaces)
        df['output'] = df['output'].apply(TextProcessor.remove_extra_spaces)
        df['input'] = df['input'].str.strip()
        df['output'] = df['output'].str.strip()
        
        return df
        

    def split_df(df, test_size=0.2, random_state=42):
        
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
        train_data.reset_index(drop=True, inplace=True)
        
        test_data.reset_index(drop=True, inplace=True)
        
        return train_data, test_data

class CharTokenizer:
    
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self, column, data):
        
        self.char2idx = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx2char = [self.PAD_TOKEN, self.UNK_TOKEN]
        
        self.column = column
        self.data = data
        
        self.__build_vocab()  # Build vocabulary

    def tokenize(self, text):
        return [char if char in self.char2idx else self.UNK_TOKEN for char in text]

    def encode(self, text):
        return [self.char2idx[char] if char in self.char2idx else self.char2idx[self.UNK_TOKEN] for char in text]
    
    def decode(self, indices):
        return ''.join([self.idx2char[idx] if idx < len(self.idx2char) else self.UNK_TOKEN for idx in indices])

    def __add_char(self, char):
        if char not in self.char2idx:
            self.idx2char.append(char)
            self.char2idx[char] = len(self.idx2char) - 1
        return self.char2idx[char]

    def __build_vocab(self):
        for sent in self.data[self.column]:
            for char in sent:
                self.__add_char(char)

class CharDataset(Dataset):
    
    def __init__(self, data, input_tokenizer, output_tokenizer):

        self.data = data
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer

    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, idx):
        
        item = self.data.iloc[idx]
        input_ids = self.input_tokenizer.encode(item['input'])
        target_ids = self.output_tokenizer.encode(item['output'])
        
        return input_ids, target_ids

def char_collate_fn(batch):
  
    lengths = [len(b[0]) for b in batch]
    max_len = max(lengths)

    sents = [b[0] + [0] * (max_len - len(b[0])) for b in batch]
    labels = [b[1] + [0] * (max_len - len(b[1])) for b in batch]

    return torch.LongTensor(sents), torch.LongTensor(labels)


