from prediction_api.predictor import Predictor

class KickstarterPitchOutcomePredictor(Predictor):

    PICKLES_PATH = Predictor.PICKLES_PATH + 'kickstarter_pitch_outcome/'
    
    def __init__(self):
        self.vectorizer =  self.load_pickle('kickstarter_vectorizer.pkl')
        self.model = self.load_pickle('kickstarter_model.pkl')
    
    def predict(self, content):
        pitch = content['pitch']
        pitch_vectorized = self.vectorizer.transform([pitch]).toarray()
        prediction = self.model.predict(pitch_vectorized)[0].item()
        return prediction