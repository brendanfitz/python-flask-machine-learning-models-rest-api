from os import path
from pandas import DataFrame
from prediction_api.predictor import Predictor

class LendingClubLoanDefaultPredictor(Predictor):

    PICKLES_PATH = Predictor.PICKLES_PATH + 'lending_club/'

    EMP_LENGTH_MAP = {
        '1': 0,
        '10plus': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5,
        '6': 6,
        '7': 7,
        '8': 8,
        '9': 9,
        'lt1': 10,
        'not_provided': 11
    }

    TERM_MAP = {'36': 0, '60': 1}

    PURPOSE_MAP =  {
        'car': 0,
        'credit_card': 1,
        'debt_consolidation': 2,
        'educational': 3,
        'home_improvement': 4,
        'house': 5,
        'major_purchase': 6,
        'medical': 7,
        'moving': 8,
        'other': 9,
        'renewable_energy': 10,
        'small_business': 11,
        'vacation': 12,
        'wedding': 13
    }
    GRADE_MAP = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6}
    
    def __init__(self):
        self.rf = self.load_pickle('rf.pkl')
    
    def predict(self, args):
        row = DataFrame(
            {
                'loan_amnt': args.get('loan_amnt'),
                'int_rate': args.get('int_rate'),
                'annual_inc': args.get('annual_inc'),
                'dti': args.get('dti'),
                'emp_length': self.EMP_LENGTH_MAP[args.get('emp_length')],
                'term': self.TERM_MAP[args.get('term')],
                'purpose': self.PURPOSE_MAP[args.get('purpose')],
                'grade': self.GRADE_MAP[args.get('grade')],
            },
            index=[0]
        )
        prediction = self.rf.predict_proba(row)[0, 1]
        return prediction