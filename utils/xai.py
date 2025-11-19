from typing import Union
import json
import numpy as np


class XAIDistanceBenchmark(object):
    
    def __init__(self, data_or_path:Union[str, list]):
        if isinstance(data_or_path, list):
            self._data = data_or_path
        else:
            with open(data_or_path, 'r') as _data_file:
                self._data = json.load(_data_file)
    
    def __len__(self):
        return len(self._data)
    
    def _fetch_distance(self, source:str, targets:list[str]):
        _distances = [
            np.stack([
                _sample['distance'][source][_target]
                for _target in targets
            ]).min(axis=0)
            for _sample in self._data
        ]
        return _distances
    
    def _fetch_distance_matrix(self, source:str, targets:list[str]):
        _distances = [
            np.stack([
                _sample['distance'][source][_target]
                for _target in targets
            ])
            for _sample in self._data
        ]
        return _distances
    
    def _fetch_index(self, source:str):
        return [_sample['index'][source] for _sample in self._data]

    def _decode_distance_index(self, key):
        
        _source, _target = key.split('->')
        _targets = _target.split(',')
        _source = _source.strip()
        _targets = [_target.strip() for _target in _targets]
        return _source, _targets
    
    def __getitem__(self, key):
        if isinstance(key, str):
            if key[0] == '*':
                key = key[1:]
                _source, _targets = self._decode_distance_index(key)
                return self._fetch_distance_matrix(_source, _targets)
                
            else:
                _source, _targets = self._decode_distance_index(key)
                return self._fetch_distance(_source, _targets)
        else:
            _source, _targets = key
            return self._fetch_distance(_source, _targets)

# Posthoc Distance Metrics
def binding_region_hit_rate(y_trues, y_preds, threshold=0.8, aggregate='mean'):
    _hit_rates = []
    for pred, gt in zip(y_preds, y_trues):
        try:
            gt = np.array(gt)
            _threshold_num = int(np.ceil(threshold * len(pred)))
            _indices = np.argsort(pred)[::-1]
            _indices = _indices[:_threshold_num]
            _gt_quantile = np.quantile(gt, 1-threshold)
            _hit_rate = np.mean(gt[_indices] < _gt_quantile)
            _hit_rates.append(_hit_rate)
        except:
            _hit_rates.append(0)
    _hit_rates = np.array(_hit_rates)
    if aggregate is None or aggregate == 'none':
        return _hit_rates
    else:
        return np.mean(_hit_rates), np.std(_hit_rates)