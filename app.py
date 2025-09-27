from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from model import MFRecommender
from preprocessing import load_data
from recommendation import load_metadata, recommend_for_new_user, recommend_for_new_user_simple

app = Flask(__name__)
CORS(app)

### Loading data and model when initalizing
print("Loading data and model...")
device = "cpu"

# Load dataset
data_bundle = load_data("../data/ratings_mini.csv")

movie2idx = data_bundle["movie2idx"]

# Load model
model = MFRecommender(
    num_users=data_bundle["num_users"],
    num_items=data_bundle["num_movies"],
    embedding_dim=32
)
model.load_state_dict(torch.load("mf_model_mini.pt", map_location=device))
model.to(device)
model.eval()

# Load metadata
movieid2title, idx2movie = load_metadata("../data/movies_mini.csv", movie2idx)
print("Model and metadata loaded.")


### API ROUTES
@app.route("/recommend", methods=["POST"])
def recommend_route():
    data = request.get_json()
    ratings_dict = data.get("ratings", {})
    top_n = int(data.get("top_n", 10))
    min_year = data.get("min_year")
    max_year = data.get("max_year")

    if not ratings_dict:
        return jsonify({"error": "No ratings provided"}), 400

    user_ratings_idx = {
        movie2idx[int(m)]: r for m, r in ratings_dict.items() if int(m) in movie2idx
    }
    if not user_ratings_idx:
        return jsonify({"error": "No valid movie IDs provided"}), 400

    recommendations = recommend_for_new_user(
        model,
        user_ratings_idx,
        movieid2title,
        idx2movie,
        top_n=top_n,
        min_year=min_year,
        max_year=max_year
    )
    return jsonify({"recommendations": recommendations})


@app.route("/recommend-simple", methods=["POST"])

def recommend_simple_route():
    data = request.get_json()
    ratings_dict = data.get("ratings", {})
    top_n = int(data.get("top_n", 10))
    min_year = data.get("min_year")
    max_year = data.get("max_year")

    if not ratings_dict:
        return jsonify({"error": "No ratings provided"}), 400
    elif len(ratings_dict) < 3:
        return jsonify({"error": "Not enough ratings provided"}), 400

    user_ratings_idx = {
        movie2idx[int(m)]: r for m, r in ratings_dict.items() if int(m) in movie2idx
    }
    if not user_ratings_idx:
        return jsonify({"error": "No valid movie IDs provided"}), 400

    recommendations = recommend_for_new_user_simple(
        user_ratings_idx,
        movieid2title,
        idx2movie,
        data_bundle["df"],
        top_n=top_n,
        min_year=min_year,
        max_year=max_year
    )
    return jsonify({"recommendations": recommendations})

@app.route("/search", methods=["GET"])
def search_movies():
    query = request.args.get("q", "").lower()
    limit = int(request.args.get("limit", 10))

    if not query:
        return jsonify([])

    # Filter movie titles
    results = [
        {"movieId": movie_id, "title": title}
        for movie_id, title in movieid2title.items()
        if query in title.lower()
    ][:limit]

    return jsonify(results)

# Start server
if __name__ == "__main__":
    print(f"Using device: {device}")
    app.run()
