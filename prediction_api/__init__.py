import numpy as np
from flask import Flask, request, jsonify
from prediction_api.predictors import (
    MovieRoiPredictor, 
    LendingClubLoanDefaultPredictor,
    KickstarterPitchOutcomePredictor,
    TitanicPredictor,
    NhlPlayerSeasonScoringTotalPredictor,
)

app = Flask(__name__)

movie_mod = MovieRoiPredictor()
loan_mod = LendingClubLoanDefaultPredictor()
kickstarter_mod = KickstarterPitchOutcomePredictor()
titanic_mod = TitanicPredictor()
nhl_mod = NhlPlayerSeasonScoringTotalPredictor()

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
