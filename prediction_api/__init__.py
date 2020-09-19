import numpy as np
from flask import Flask, request, jsonify
from prediction_api.movie_roi_predictor import MovieRoiPredictor 
from prediction_api.lending_club_loan_default_predictor import LendingClubLoanDefaultPredictor

app = Flask(__name__)

movie_mod = MovieRoiPredictor()
loan_mod = LendingClubLoanDefaultPredictor()

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

