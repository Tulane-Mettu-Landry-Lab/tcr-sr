from proteinbert import load_pretrained_model
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

import json
import torch
import glob
import os
from tqdm import tqdm
from itertools import chain

def obtain_proteinbert_batch(batch, model, max_len=43):
    encoded_x = input_encoder.encode_X(batch, max_len)
    local_representations, global_representations = model.predict(encoded_x, batch_size=len(batch))
    mask = torch.tensor(encoded_x[0] != 25, dtype=torch.long)
    hidden_states = torch.tensor(local_representations)
    logits = torch.tensor(global_representations)
    return logits, hidden_states, mask

def obtain_proteinbert(data, model, max_len=45, save_path='./', batchsize=256):
    os.makedirs(save_path, exist_ok=True)
    print(len(data))
    for bi, i in enumerate(tqdm(range(0, len(data), batchsize))):
        batch = data[i:i+batchsize]
        logits, hidden_states, mask = obtain_proteinbert_batch(batch, model, max_len=max_len)
        torch.save(hidden_states, os.path.join(save_path, f'{bi}_hidden_states.pt'))
        torch.save(logits, os.path.join(save_path, f'{bi}_logits.pt'))
        torch.save(mask, os.path.join(save_path, f'{bi}_attention_mask.pt'))

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

if __name__ == '__main__':
    save_path = 'embeddings/proteinbert'
    batchsize = 128
    data_path = 'data/index.json'
    with open(data_path, 'r') as f:
        _data_index = json.load(f)
    data = list(chain(*_data_index.values()))
    max_length = max([len(i) for i in data])+2

    pretrained_model_generator, input_encoder = load_pretrained_model()
    model = get_model_with_hidden_layers_as_outputs(pretrained_model_generator.create_model(max_length))
    
    obtain_proteinbert(
        data, model,
        max_len=max_length,
        save_path=save_path,
        batchsize=batchsize
    )
    process_embeddings_chunk(save_path)
    with open(os.path.join(save_path, 'index.json'), 'w') as f:
        json.dump(_data_index, f, indent=2)
        
    
    save_path = 'embeddings/tcrxai/proteinbert'
    batchsize = 128
    data_path = 'data/TCRXAI/index.json'
    with open(data_path, 'r') as f:
        _data_index = json.load(f)
    data = list(chain(*_data_index.values()))
    max_length = max([len(i) for i in data])+2

    pretrained_model_generator, input_encoder = load_pretrained_model()
    model = get_model_with_hidden_layers_as_outputs(pretrained_model_generator.create_model(max_length))
    
    obtain_proteinbert(
        data, model,
        max_len=max_length,
        save_path=save_path,
        batchsize=batchsize
    )
    process_embeddings_chunk(save_path)
    with open(os.path.join(save_path, 'index.json'), 'w') as f:
        json.dump(_data_index, f, indent=2)
    
    