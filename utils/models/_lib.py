class ModelLib(object):
    _models = {}
    @classmethod
    def register(cls, model, name=None):
        if name is None:
            cls._models[model.__qualname__] = model
        else:
            cls._models[name] = model
    
    @classmethod
    def models(cls):
        return list(cls._models.keys())
    
    def __class_getitem__(cls, key):
        return cls._models[key]