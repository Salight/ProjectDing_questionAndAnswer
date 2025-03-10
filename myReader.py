import torch
import json
from torch.utils.data import Dataset, DataLoader

class DuReaderQG(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                one_data = json.loads(line.strip())
                Data[idx] = {
                    'question': "给定上下文：{} 。回答问题：{}".format(one_data['context'],one_data['question']), 
                    'answer': one_data['answer'],
                }
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
        
def get_dataLoader(dataset, model, tokenizer, max_input_length, max_target_length, batch_size=32, shuffle=False):
    def collote_fn(batch_samples):
        batch_questions, batch_contexts, batch_answers = [], [], []
        for sample in batch_samples:
            batch_questions.append(sample['question'])
            batch_answers.append(sample['answer'])
    
        batch_data = tokenizer(
            batch_questions,
            padding=True, 
            max_length=max_input_length,
            truncation=True, 
            return_tensors="pt"
        )
        with tokenizer.as_target_tokenizer():
            tokenized = tokenizer(
                batch_answers, 
                padding=True, 
                max_length=max_target_length,
                truncation=True, 
                return_tensors="pt"
            )["input_ids"]
            labels = tokenized[:,1:]
            # batch_data['decoder_input_ids'] = tokenized[:,:-1]
            batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)
            end_token_index = torch.where(labels == tokenizer.sep_token_id)[1]
            for idx, end_idx in enumerate(end_token_index):
                labels[idx][end_idx+1:] = -100
            batch_data['labels'] = labels
        if 'token_type_ids' in batch_data:
            del batch_data['token_type_ids']
        return batch_data
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                      collate_fn=collote_fn)

