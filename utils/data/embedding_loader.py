import json
import os
import torch
import pandas as pd
import numpy as np
from itertools import chain
from concurrent.futures import ThreadPoolExecutor


class EmbeddingLoader(object):

    def __init__(
        self,
        weights_path = 'embeddings/esm3_open',
        index_json_path = 'data/Full/full_np_index.json',
        batch_size = 256,
        max_workers = 10,
        cache = True
    ):
        self.weights_path = weights_path
        self.index_json_path = index_json_path
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.index_table = self.create_index_table(
            index_json_path=index_json_path,
            batch_size=batch_size
        )
        self.cache = cache
        self.cache_embeddings = {}


    def create_index_table(self, index_json_path='data/Full/full_np_index.json', batch_size=256):
        with open(index_json_path, 'r') as f:
            index_aa = json.load(f)
        df = pd.DataFrame(list(chain(*index_aa.values())), columns=['aa'])
        df = df.reset_index()
        df['page'] = df['index'] // batch_size
        df['index'] = df['index'] % batch_size
        return df
    
    def fetch_weights_inpage(self, weights_path, name='hidden_states', page=0, index=[0, 1, 2]):
        
        _path = os.path.join(weights_path, f'{page}_{name}.pt')
        _weight = torch.load(_path, map_location='cpu')
        self._buffer[page] = _weight[index]
        # return _weight[index]
    

    def fetch_weights(self, weights_path, name='hidden_states', page_indexs=[]):
        page_indexs = np.array(sorted(set(page_indexs)))
        pages = np.unique(page_indexs.T[0])
        queries = {page:page_indexs.T[1][page_indexs.T[0] == page] for page in pages}
        data = None
        tasks = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            self._buffer = {}
            for page, index in queries.items():
                tasks.append(pool.submit(self.fetch_weights_inpage, weights_path, name, page, index))
            pool.shutdown()
            for _buff_id in sorted(self._buffer):
                if data is None:
                    data = self._buffer[_buff_id]
                else:
                    data = torch.concat([
                        data,
                        self._buffer[_buff_id]
                    ])
        return page_indexs, data
    
    def fetch_weights_byseqs(self, seqs, name='hidden_states'):
        index_table = self.index_table.merge(pd.Series(seqs, name='aa'), on='aa', how='inner')
        page_index = list(map(tuple, index_table[['page', 'index']].to_numpy().tolist()))
        aa_seqs = index_table['aa'].values

        page_indexs, data = self.fetch_weights(weights_path=self.weights_path, name=name, page_indexs=page_index)

        page_index = dict(zip(page_index, aa_seqs))
        page_indexs = list(map(tuple, page_indexs))
        seq_orders = [page_index[pi] for pi in page_indexs]
        
        return seq_orders, data