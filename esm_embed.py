from transformers import AutoTokenizer, EsmForMaskedLM
from accelerate import Accelerator
import torch
import torch.nn.functional as F
import json
import os
import glob
from itertools import chain
from tqdm import tqdm

@torch.no_grad
def embed_aa(aa, tokenizer=None, model=None, device='cuda', max_length=44):
    inputs = tokenizer(aa, return_tensors="pt", padding='max_length', max_length=max_length, truncation=True).to(device)
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]
    logits = model.lm_head(hidden_states)
    hidden_states = hidden_states.cpu()
    logits = logits.cpu()
    attention_mask = inputs['attention_mask'].cpu()
    return hidden_states, logits, attention_mask


def get_embeddings(data, model, tokenizer, save_path, batch_size=128, device='cuda', max_length=44):
    hidden_states_s, logits_s, attention_mask_s = None, None, None
    for bi, i in tqdm(enumerate(range(0, len(data), batch_size))):
        batch = data[i:i+batch_size]
        hidden_states, logits, attention_mask = \
            embed_aa(
                batch,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_length=max_length
            )
        torch.save(hidden_states, os.path.join(save_path, f'{bi}_hidden_states.pt'))
        torch.save(logits, os.path.join(save_path, f'{bi}_logits.pt'))
        torch.save(attention_mask, os.path.join(save_path, f'{bi}_attention_mask.pt'))

def get_data(path, file_sel='attention_mask', size=0):
    data = None
    for i in range(size):
        _data_chunk = torch.load(os.path.join(path, f'{i}_{file_sel}.pt'), map_location='cpu')
        if data is None:
            data = _data_chunk
        else:
            data = torch.concat([data, _data_chunk])
    return data
def remove_chunks(path, file_sel='attention_mask'):
    for filepath in glob.glob(os.path.join(path, f'*_{file_sel}.pt')):
        os.remove(filepath)

def save_ckpt(weight, path, file_sel='attention_mask'):
    torch.save(weight, os.path.join(path, f'{file_sel}.pt'))
    
def process_one_file(path, file_sel='attention_mask', size=0):
    _data = get_data(path, file_sel, size)
    remove_chunks(path, file_sel)
    save_ckpt(_data, path, file_sel)
    
def process_embeddings_chunk(path):
    files = os.listdir(path)
    size = len(files) // 3
    for i in ['attention_mask', 'hidden_states', 'logits']:
        process_one_file(path, i, size)

def obtain_esm_embedding(model_path, data_path, save_path, batch_size=128, device='cuda'):
    with open(data_path, 'r') as f:
        _data_index = json.load(f)
    data = list(chain(*_data_index.values()))
    max_length = max([len(i) for i in data])+2

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = EsmForMaskedLM.from_pretrained(model_path)
    
    os.makedirs(save_path, exist_ok=True)

    model = model.to(device)

    get_embeddings(
        data = data,
        model = model,
        tokenizer = tokenizer,
        save_path = save_path,
        batch_size = batch_size,
        device = device,
        max_length = max_length
    )
    process_embeddings_chunk(save_path)
    with open(os.path.join(save_path, 'index.json'), 'w') as f:
        json.dump(_data_index, f, indent=2)

if __name__ == '__main__':
    data_path = 'data/index.json'
    batch_size = 128
    device='cuda:0'
    save_path = 'embeddings/'
    models = {
        'esm1b_t33_650M_UR50S': 'facebook/esm1b_t33_650M_UR50S',
        'esm2_t6_8M_UR50D': 'facebook/esm2_t6_8M_UR50D',
        'esm2_t12_35M_UR50D': 'facebook/esm2_t12_35M_UR50D',
        'esm2_t30_150M_UR50D': 'facebook/esm2_t30_150M_UR50D',
        'esm2_t33_650M_UR50D': 'facebook/esm2_t33_650M_UR50D',
        'esm2_t36_3B_UR50D': 'facebook/esm2_t36_3B_UR50D',
    }
    for model_name, model_path in models.items():
        obtain_esm_embedding(
            model_path=model_path,
            data_path=data_path,
            save_path=os.path.join(save_path, model_name),
            batch_size=batch_size,
            device=device
        )
        
    
    data_path = 'data/TCRXAI/index.json'
    batch_size = 128
    device='cuda:0'
    save_path = 'embeddings/tcrxai/'
    models = {
        'esm1b_t33_650M_UR50S': 'facebook/esm1b_t33_650M_UR50S',
        'esm2_t6_8M_UR50D': 'facebook/esm2_t6_8M_UR50D',
        'esm2_t12_35M_UR50D': 'facebook/esm2_t12_35M_UR50D',
        'esm2_t30_150M_UR50D': 'facebook/esm2_t30_150M_UR50D',
        'esm2_t33_650M_UR50D': 'facebook/esm2_t33_650M_UR50D',
        'esm2_t36_3B_UR50D': 'facebook/esm2_t36_3B_UR50D',
    }
    for model_name, model_path in models.items():
        obtain_esm_embedding(
            model_path=model_path,
            data_path=data_path,
            save_path=os.path.join(save_path, model_name),
            batch_size=batch_size,
            device=device
        )