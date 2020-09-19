import numpy as np
from flask import Flask, request, jsonify
from prediction_api.movie_roi_predictor import MovieRoiPredictor 
from prediction_api.lending_club_loan_default_predictor import LendingClubLoanDefaultPredictor
from prediction_api.kickstarter_pitch_outcome import KickstarterPitchOutcomePredictor 
from prediction_api.titanic_predictor import TitanicPredictor
from prediction_api.nhl_scoring_total import NhlPlayerSeasonScoringTotal

app = Flask(__name__)

movie_mod = MovieRoiPredictor()
loan_mod = LendingClubLoanDefaultPredictor()
kickstarter_mod = KickstarterPitchOutcomePredictor()
titanic_mod = TitanicPredictor()
nhl_mod = NhlPlayerSeasonScoringTotal()

@app.route('/')
def hello():
    return "Hello World!"

@app.route('/movie_roi')
def movie_roi():
    prediction = movie_mod.predict(request.args)
    data = {
        'prediction': prediction,
    }
    return jsonify(data)

@app.route('/lending_club_loan_default')
def lending_club_loan_default():
    prediction = loan_mod.predict(request.args)
    data = {
        'prediction': prediction,
    }
    return jsonify(data)

@app.route('/kickstarter_pitch_outcome', methods=['POST'])
def kickstarter_pitch_outcome():
    content = request.get_json()
    prediction = kickstarter_mod.predict(content)
    data = {
        'prediction': prediction,
    }
    return jsonify(data)

@app.route('/titanic')
def titanic():
    prediction = titanic_mod.predict(request.args)
    data = {
        'prediction': prediction,
    }
    return jsonify(data)

@app.route('/nhl_player_season_scoring_total')
def nhl_player_season_scoring_total():
    prediction = nhl_mod.predict(request.args)
    data = {
        'prediction': prediction,
    }
    return jsonify(data)
