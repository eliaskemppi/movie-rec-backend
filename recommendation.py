# recommendation.py
import torch
import re
import pandas as pd

# Get 
def load_metadata(movie_csv_path, movie2idx):
    movies_df = pd.read_csv(movie_csv_path)
    movieid2title = dict(zip(movies_df["movieId"], movies_df["title"]))
    idx2movie = {i: m for m, i in movie2idx.items()}
    return movieid2title, idx2movie

def extract_year(title: str):
    match = re.search(r"\((\d{4})\)", title)
    return int(match.group(1)) if match else None

def recommend_for_new_user(
    model,
    user_ratings_dict,
    movieid2title,
    idx2movie,
    top_n=10,
    min_year=None,
    max_year=None,
    epochs=100,
    lr=0.01
):
    device = "cpu"

    # Freeze model weights
    for param in model.parameters():
        param.requires_grad = False

    # Train new user embedding
    torch.manual_seed(99)
    user_embed = torch.randn(1, model.user_emb.embedding_dim, requires_grad=True, device=device)
    movie_ids_tensor = torch.tensor(list(user_ratings_dict.keys()), device=device)
    movie_embeds = model.item_emb(movie_ids_tensor)
    ratings_tensor = torch.tensor(list(user_ratings_dict.values()), dtype=torch.float, device=device)

    optimizer = torch.optim.Adam([user_embed], lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        preds = (user_embed * movie_embeds).sum(dim=1)
        loss = loss_fn(preds, ratings_tensor)
        loss.backward()
        optimizer.step()

    # Score all movies
    all_movie_embeds = model.item_emb.weight
    scores = (user_embed @ all_movie_embeds.T).squeeze()

    # Filter out already rated movies
    already_rated = set(user_ratings_dict.keys())
    mask = torch.tensor([i not in already_rated for i in range(all_movie_embeds.shape[0])], device=device)

    # Filter by year if provided
    if min_year is not None or max_year is not None:
        year_mask = []
        for i in range(all_movie_embeds.shape[0]):
            movie_id = idx2movie[i]
            title = movieid2title.get(movie_id, "")
            year = extract_year(title)
            if year is None:
                year_mask.append(True)
            else:
                if min_year is not None and year < min_year:
                    year_mask.append(False)
                    continue
                if max_year is not None and year > max_year:
                    year_mask.append(False)
                    continue
                year_mask.append(True)
        year_mask = torch.tensor(year_mask, device=device)
        mask = mask & year_mask

    scores_filtered = scores[mask]
    movie_ids_filtered = torch.arange(all_movie_embeds.shape[0], device=device)[mask]

    top_indices = torch.topk(scores_filtered, top_n).indices.cpu().numpy()
    recommended_movie_ids = [movie_ids_filtered[i].item() for i in top_indices]
    return [movieid2title[idx2movie[mid]] for mid in recommended_movie_ids]

def recommend_for_new_user_simple(
    user_ratings_dict,
    movieid2title,
    idx2movie,
    ratings_df,
    top_n=10,
    min_year=None,
    max_year=None,
    k=20,
    min_raters=3
):
    
    # Make a copy so we don’t mutate the shared DataFrame
    df = ratings_df.copy()

    # Add the new user (use an id not in dataset, e.g. 999)
    new_user_id = max(df["userId"].max() + 1, 999)

    # Add the new user data 
    new_user_ratings = pd.DataFrame([
        {"userId": new_user_id, "movieId": mid, "rating": r} 
        for mid, r in user_ratings_dict.items()
    ])
    df = pd.concat([df, new_user_ratings], ignore_index=True)

    # Build user-item matrix (users as rows and movies as columns)
    user_item_matrix = df.pivot_table(
        index="userId", columns="movieId", values="rating"
    )

    # Build a user similarity matrix based on the movie ratings. (rows and cols are users, values are how similar they are)
    user_sim = user_item_matrix.T.corr(method="pearson", min_periods=3)

    # Find k most similar users
    similar_users = (
        user_sim[new_user_id]
        .drop(index=new_user_id)
        .loc[lambda x: x > 0]
        .dropna() 
        .sort_values(ascending=False)
        .head(k)
)

    #print(similar_users)
    # Pick the movies the new user has not seen
    unseen_movies = user_item_matrix.columns[user_item_matrix.loc[new_user_id].isna()]

    predictions = {}

    # Iterate trough the unseen movies, compute the weighted average rating for the k most similar users
    for mid in unseen_movies:
        relevant_ratings = user_item_matrix.loc[similar_users.index, mid].dropna()
        if len(relevant_ratings) < min_raters:
            continue

        sims = similar_users[relevant_ratings.index]
        
        pred_rating = (relevant_ratings * sims).sum() / sims.sum()
        predictions[mid] = pred_rating.round(3)

    # Sort and convert to DataFrame
    rec_df = pd.DataFrame(
        sorted(predictions.items(), key=lambda x: x[1], reverse=True),
        columns=["reindexed_id", "predicted_rating"]
    )

    # Map back: reindexed_id -> original movieId
    rec_df["movieId"] = rec_df["reindexed_id"].map(idx2movie)

    # Map movieId -> title
    rec_df["title"] = rec_df["movieId"].map(movieid2title)

    # Apply year filters if needed
    if min_year is not None or max_year is not None:
        import re
        def extract_year(title):
            match = re.search(r"\((\d{4})\)", str(title))
            return int(match.group(1)) if match else None

        rec_df["year"] = rec_df["title"].apply(extract_year)
        if min_year is not None:
            rec_df = rec_df[rec_df["year"] >= min_year]
        if max_year is not None:
            rec_df = rec_df[rec_df["year"] <= max_year]

    return rec_df["title"].head(top_n).tolist()
