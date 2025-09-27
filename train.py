import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from preprocessing import load_data
from model import MFRecommender

data_bundle = load_data("../data/ratings_mini.csv")

train = data_bundle["train"]
test = data_bundle["test"]
num_users = data_bundle["num_users"]
num_movies = data_bundle["num_movies"]


class RatingsDataset(Dataset):
    def __init__(self, df):
        self.user_ids = torch.tensor(df["userId"].values, dtype=torch.long)
        self.movie_ids = torch.tensor(df["movieId"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float)
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.user_ids[idx], self.movie_ids[idx], self.ratings[idx]

train_dataset = RatingsDataset(train)
test_dataset = RatingsDataset(test)

batch_size = 2048

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


device = "cpu"

def test(model, test_loader, loss_fn, device=device):
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for user_ids, movie_ids, ratings in test_loader:
            user_ids = user_ids.to(device)
            movie_ids = movie_ids.to(device)
            ratings = ratings.to(device)

            preds = model(user_ids, movie_ids)
            loss = loss_fn(preds, ratings)

            total_loss += loss.item() * len(ratings)
            count += len(ratings)

    avg_loss = total_loss / count
    return avg_loss


def main():

    model = MFRecommender(
        num_users=num_users,
        num_items=num_movies,
        embedding_dim=32
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    num_epochs = 200
    for epoch in range(num_epochs):
        # ---- training ----
        model.train()
        total_loss = 0

        for user_ids, movie_ids, ratings in train_loader:
            user_ids = user_ids.to(device)
            movie_ids = movie_ids.to(device)
            ratings = ratings.to(device)

            optimizer.zero_grad()
            preds = model(user_ids, movie_ids)
            loss = loss_fn(preds, ratings)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(ratings)

        avg_train_loss = total_loss / len(train_dataset)
        train_rmse = avg_train_loss ** 0.5

        # ---- evaluation ----
        avg_test_loss = test(model, test_loader, loss_fn, device)
        test_rmse = avg_test_loss ** 0.5

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")

    torch.save(model.state_dict(), "mf_model_mini.pt")
    print("Model saved to mf_model_mini.pt")


if __name__ == "__main__":
    main()
