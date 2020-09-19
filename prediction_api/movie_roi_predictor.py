from os import path
from prediction_api.predictor import Predictor

class MovieRoiPredictor(Predictor):

    PICKLES_PATH = Predictor.PICKLES_PATH + 'movie_predictor/'
    
    def __init__(self):
        self.regr = self.load_pickle('luther_model.pkl')
        self.budget_poly = self.load_pickle('budget_poly.pkl')
        self.budget_poly_scaler = self.load_pickle('budget_poly_scaler.pkl')
        self.ohe = self.load_pickle('ohe.pkl')
        self.cv = self.load_pickle('cv.pkl')
        self.passthroughs_scaler = self.load_pickle('passthroughs_scaler.pkl')