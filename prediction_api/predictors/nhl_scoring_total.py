from pandas import DataFrame
from os import path
from statsmodels.regression.linear_model import OLSResults
from prediction_api.predictors.predictor import Predictor

class NhlPlayerSeasonScoringTotalPredictor(Predictor):

    PICKLES_PATH = Predictor.PICKLES_PATH + 'nhl/'

    def __init__(self):
        self.model = self.load_model('nhl_goals_regression_model.pkl')
    
    def load_model(self, filename):
        filepath = path.join(self.PICKLES_PATH, filename)
        with open(filepath, 'rb') as f:
            model = OLSResults.load(f)
        return model
    
    def predict(self, args):
        row = DataFrame(
            {
                'L1': int(args.get('l1')),
                'L2': int(args.get('l2')),
                'L3': int(args.get('l3')),
                'L4': int(args.get('l4')),
                'L5': int(args.get('l5')),
                'season_number': int(args.get('season_number')),
                'gamesPlayed': int(args.get('games_played')),
                'positionCode': args.get('position').upper(),
            }, index=[0]
        )
        prediction = self.model.predict(row)[0].item()
        return prediction