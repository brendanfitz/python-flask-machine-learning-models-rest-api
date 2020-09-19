from pandas import DataFrame
from prediction_api.predictors.predictor import Predictor

class TitanicPredictor(Predictor):

    PICKLES_PATH = Predictor.PICKLES_PATH + 'titanic/'

    SEX_MAP = {'male': 0, 'female': 1}
    EMBARKED_MAP = {'c': 0, 'q': 1, 's': 2}
    TITLE_MAP = {
       'army': 0,
       'master': 1,
       'medical': 2,
       'miss': 3,
       'mr': 4,
       'mrs': 5,
       'navy': 6,
       'rare': 7,
       'religious': 8
    }

    def __init__(self):
        self.model =  self.load_pickle('titanic_model.pkl')
    
    def predict(self, args):
        family_size = int(args.get('family_size'))
        age = int(args.get('age'))
        row = DataFrame(
            {
                'pclass': int(args.get('pclass')),
                'sex': self.SEX_MAP[args.get('sex')],
                'title': self.TITLE_MAP[args.get('title')],
                'embarked': self.EMBARKED_MAP[args.get('embarked')],
                'family_size': family_size,
                'is_alone': self.is_alone(family_size),
                'age_category': self.age_label(age),
            },
            index=[0]
        )
        prediction = self.model.predict(row)[0].item()
        return prediction

    @staticmethod
    def age_label(age):
        if age < 12:
            return 4
        elif age < 18:
            return 2
        elif age < 50:
            return 1
        elif age < 100:
            return 3
        else:
            return 

    @staticmethod
    def is_alone(family_size):
        if family_size > 0:
            return 0
        return 1