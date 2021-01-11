import os
import json
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import PIL
from PIL import Image
from transformers import T5Tokenizer
from transformers import DistilBertTokenizer

tokenizer = T5Tokenizer.from_pretrained('t5-base')
#Pre treinado no squad
bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Normalization suggested by the EfficientNet library when trained with adversarial examples.
# https://github.com/lukemelas/EfficientNet-PyTorch#update-january-23-2020
transform = transforms.Compose([
    transforms.Lambda(lambda img: img * 2.0  - 1.0),
])

class DocVQADataset(Dataset):
    """
    Dataset that loads image instances lazily to memory.
    """

    def __init__(self, mode, path, img_transform=lambda x: x):
        
        self.mode = mode
        self.path = path
        self.height = 700
        self.width = 400
        
        self.img_transform = img_transform
        
        with open(f"/data/{self.mode}/{self.mode}_v1.0.json", 'r') as data_json_file:
                self.data = json.load(data_json_file)['data']

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        file_infos = self.data[idx]
        question = file_infos['question']
        answers = file_infos['answers'] if self.mode in ['train', 'val'] else []
        
        img_file = file_infos['image']
        img = Image.open(f"{self.path}/{img_file}")
        img = img.resize((self.width, self.height))
        img = self.img_transform(img)

        img = np.array(img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = np.moveaxis(img, 2, 0) 
            
        # Convert to float.
        img = img.astype(float)

        # Normalize image between 0 and 1.
        img = (img - img.min()) / np.max([img.max() - img.min(), 1])
                
        return transform(img), question, answers

class OCRDataset(Dataset):
    """
    Dataset that loads image instances lazily to memory.
    """

    def __init__(self, mode, path):
        
        self.mode = mode
        self.path = path
        
        with open(f"/data/{self.mode}/{self.mode}_v1.0.json", 'r') as data_json_file:
                self.data = json.load(data_json_file)['data']

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        file_infos = self.data[idx]
        question = 'Question: ' + file_infos['question']
        answers = file_infos['answers'] if self.mode in ['train', 'val'] else []
        document_id = file_infos['ucsf_document_id'] + '_' + file_infos['ucsf_document_page_no']
        
        with open(f"{self.path}/ocr_results/{document_id}.json", 'r') as data_json_file:
                r_results = json.load(data_json_file)['recognitionResults']
        lines = r_results[0]['lines']
        ocr_results = ' '.join([ l['text'] for l in lines])
        ocr_results = 'Context: ' + ocr_results
        
        input_text = question + tokenizer.eos_token + ocr_results
        
        return input_text, question, answers
    

def collate_any(batch):
    """
    Question sample and answers for every image.
    """
    
    imgs = [r[0] for r in batch]
    questions = [r[1] for r in batch]
    answers = [random.choice(r[2]) for r in batch]
    
    batch_tokens_questions = tokenizer.batch_encode_plus(questions, return_tensors="pt", truncation=True, max_length=32, padding='max_length')
    batch_tokens_answers = tokenizer.batch_encode_plus(answers, return_tensors="pt", truncation=True, max_length=32, padding='max_length')

    return (
        torch.tensor(imgs, dtype=torch.float), 
        batch_tokens_questions["input_ids"],
        batch_tokens_answers["input_ids"],
        answers,
    )

def collate_one(batch):
    """
    Question sample and one answer for every image.
    """
    
    imgs = [r[0] for r in batch]
    questions = [r[1] for r in batch]
    answers = [r[2][0] for r in batch]
    
    batch_tokens_questions = tokenizer.batch_encode_plus(questions, return_tensors="pt", truncation=True, max_length=32, padding='max_length')
    batch_tokens_answers = tokenizer.batch_encode_plus(answers, return_tensors="pt", truncation=True, max_length=32, padding='max_length')

    return (
        torch.tensor(imgs, dtype=torch.float), 
        batch_tokens_questions["input_ids"],
        batch_tokens_answers["input_ids"],
        answers,
    )


def collate_bert_any(batch):
    """
    Question sample and one answer for every image.
    """
    
    imgs = [r[0] for r in batch]
    questions = [r[1] for r in batch]
    answers = [random.choice(r[2]) for r in batch]
    
    batch_tokens_questions = bert_tokenizer.batch_encode_plus(questions, return_tensors="pt", truncation=True, max_length=32, padding='max_length')
    batch_tokens_answers = tokenizer.batch_encode_plus(answers, return_tensors="pt", truncation=True, max_length=32, padding='max_length')

    return (
        torch.tensor(imgs, dtype=torch.float), 
        batch_tokens_questions["input_ids"],
        batch_tokens_answers["input_ids"],
        answers,
    )

def collate_bert_one(batch):
    """
    Question sample and one answer for every image.
    """
    
    imgs = [r[0] for r in batch]
    questions = [r[1] for r in batch]
    answers = [r[2][0] for r in batch]
    
    batch_tokens_questions = bert_tokenizer.batch_encode_plus(questions, return_tensors="pt", truncation=True, max_length=32, padding='max_length')
    batch_tokens_answers = tokenizer.batch_encode_plus(answers, return_tensors="pt", truncation=True, max_length=32, padding='max_length')

    return (
        torch.tensor(imgs, dtype=torch.float), 
        batch_tokens_questions["input_ids"],
        batch_tokens_answers["input_ids"],
        answers,
    )

def collate_ocr_any(batch):
    """
    Question sample and answers for every image.
    """
    
    input_text = [r[0] for r in batch]
    questions = [r[1] for r in batch]
    answers = [random.choice(r[2]) for r in batch]

    batch_tokens_input = tokenizer.batch_encode_plus(input_text, return_tensors="pt", truncation=True, max_length=128, padding='max_length')
    batch_tokens_answers = tokenizer.batch_encode_plus(answers, return_tensors="pt", truncation=True, max_length=128, padding='max_length')

    return (
        batch_tokens_input["input_ids"], 
        batch_tokens_input["attention_mask"],
        batch_tokens_answers["input_ids"],
        questions,
        answers
    )

def collate_ocr_one(batch):
    """
    Question sample and one answer for every image.
    """
    
    input_text = [r[0] for r in batch]
    questions = [r[1] for r in batch]
    answers = [r[2][0] for r in batch]
    
    batch_tokens_input = tokenizer.batch_encode_plus(input_text, return_tensors="pt", truncation=True, max_length=128, padding='max_length')
    batch_tokens_answers = tokenizer.batch_encode_plus(answers, return_tensors="pt", truncation=True, max_length=128, padding='max_length')

    return (
        batch_tokens_input["input_ids"], 
        batch_tokens_input["attention_mask"],
        batch_tokens_answers["input_ids"],
        questions,
        answers
    )