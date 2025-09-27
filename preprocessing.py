import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(ratings_csv_path="../data/ratings_mini.csv", min_ratings_per_movie=5):
    """Load ratings data, filter low-frequency movies, and return df, mappings, and train/test splits."""
    df = pd.read_csv(ratings_csv_path).drop(columns=["timestamp"])

    # Filter movies with fewer than min_ratings_per_movie
    movie_counts = df["movieId"].value_counts()
    valid_movies = movie_counts[movie_counts >= min_ratings_per_movie].index
    df = df[df["movieId"].isin(valid_movies)]

    # Drop users who now have 0 ratings left
    user_counts = df["userId"].value_counts()
    valid_users = user_counts[user_counts > 0].index
    df = df[df["userId"].isin(valid_users)]

    # Reindex for embeddings
    user2idx = {u: i for i, u in enumerate(df["userId"].unique())}
    movie2idx = {m: i for i, m in enumerate(df["movieId"].unique())}
    df["userId"] = df["userId"].map(user2idx)
    df["movieId"] = df["movieId"].map(movie2idx)

    num_users = len(user2idx)
    num_movies = len(movie2idx)

    # Train/test split
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train_users = set(train["userId"])
    train_movies = set(train["movieId"])

    # Keep only valid test entries
    test = test[test["userId"].isin(train_users) & test["movieId"].isin(train_movies)]

    return {
        "df": df,
        "user2idx": user2idx,
        "movie2idx": movie2idx,
        "num_users": num_users,
        "num_movies": num_movies,
        "train": train,
        "test": test
    }
