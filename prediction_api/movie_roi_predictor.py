from os import path
from numpy import concatenate
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
    
    def predict(self, args):
        budget = args.get('budget')
        budget_scaled = self.budget_poly.transform([[budget]])
        budget_df = self.budget_poly_scaler.transform(budget_scaled)

        passthroughs_df = self.passthroughs_scaler.transform([
           [
               args.get('in_release_days'),
               args.get('widest_release'),
               args.get('runtime'),
           ],
        ])

        rating = args.get('rating')
        rating_df = self.ohe.transform([[rating]]).toarray()

        genre = args.get('genre')
        genre_df = self.cv.transform([genre]).toarray()

        frames = [budget_df, passthroughs_df, rating_df, genre_df]
        row = concatenate(frames, axis=1)

        prediction = self.regr.predict(row)[0]
        
        return prediction