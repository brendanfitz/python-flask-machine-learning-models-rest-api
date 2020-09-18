from os import path
import pickle
import prediction_api.luther_util as luther_util

class MovieRoiPredictor(object):

    PICKLES_PATH = './prediction_api/static/pickles/movie_predictor/'
    
    def __init__(self):
        self.regr = self.load_pickle('luther_model.pkl')
        self.budget_poly = self.load_pickle('budget_poly.pkl')
        self.budget_poly_scaler = self.load_pickle('budget_poly_scaler.pkl')
        self.ohe = self.load_pickle('ohe.pkl')
        self.cv = self.load_pickle('cv.pkl')
        self.passthroughs_scaler = self.load_pickle('passthroughs_scaler.pkl')

    def load_pickle(self, filename):
        filepath = path.join(self.PICKLES_PATH, filename)
        with open(filepath, 'rb') as f:
            pkl_obj = pickle.load(f)
        return pkl_obj