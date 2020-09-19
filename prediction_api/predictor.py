from os import path
import pickle

class Predictor(object):

    PICKLES_PATH = './prediction_api/static/pickles/'
    
    def load_pickle(self, filename):
        filepath = path.join(self.PICKLES_PATH, filename)
        with open(filepath, 'rb') as f:
            pkl_obj = pickle.load(f)
        return pkl_obj