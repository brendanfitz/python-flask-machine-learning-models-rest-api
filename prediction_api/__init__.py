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
    args = request.args
    print(args)

    budget = request.args.get('budget')
    budget_scaled = movie_mod.budget_poly.transform([[budget]])
    budget_df = movie_mod.budget_poly_scaler.transform(budget_scaled)

    passthroughs_df = movie_mod.passthroughs_scaler.transform([
       [
           request.args.get('in_release_days'),
           request.args.get('widest_release'),
           request.args.get('runtime'),
       ],
    ])

    rating = request.args.get('rating')
    rating_df = movie_mod.ohe.transform([[rating]]).toarray()

    genre = request.args.get('genre')
    genre_df = movie_mod.cv.transform([genre]).toarray()

    frames = [budget_df, passthroughs_df, rating_df, genre_df]
    row = np.concatenate(frames, axis=1)

    prediction = movie_mod.regr.predict(row)[0]

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

