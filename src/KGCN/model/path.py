import os

class Path:
    def __init__(self, dataset):
        self.data = f'../../data/{dataset}/'
        self.misc = f'../../misc/{dataset}/'
        self.emb = f'../../misc/{dataset}/emb/'
        self.output = f'../../output/KGCN/{dataset}/'
        
        self.check_dir(f'../../output/KGCN/')
        self.check_dir(self.data)
        self.check_dir(self.misc)
        self.check_dir(self.emb)
        self.check_dir(self.output)

    def check_dir(self, p):
        if not os.path.isdir(p):
            os.mkdir(p)
