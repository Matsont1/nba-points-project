# train_model.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def build_feature_table():
    players = pd.read_csv("PlayerStatistics.csv", low_memory=False)
    games = pd.read_csv("Games(1).csv", low_memory=False)
    players["gameDateTimeEst"] = pd.to_datetime(players["gameDateTimeEst"], errors="coerce")
    games["gameDateTimeEst"] = pd.to_datetime(games["gameDateTimeEst"], errors="coerce")
    players["PlayerName"] = players["firstName"] + " " + players["lastName"]
    players = players.sort_values(["PlayerName", "gameDateTimeEst"]).reset_index(drop=True)

    def last5_avg(points_series):
        return points_series.shift(1).rolling(5, min_periods=1).mean()
        players["last5_avg_points"] = (players.groupby("PlayerName")["points"].transform(last5_avg))

    season_avg = (players.groupby("PlayerName", as_index=False)["points"].mean().rename(columns={"points": "season_avg_points"}))

    long_rows = []
    for _, row in games.iterrows():
        long_rows.append({"teamName": row["hometeamName"], "pointsAllowed": row["awayScore"]})
        long_rows.append({"teamName": row["awayteamName"], "pointsAllowed": row["homeScore"]})

    team_def = pd.DataFrame(long_rows)
    team_def = (
        team_def.groupby("teamName", as_index=False)["pointsAllowed"].mean().rename(columns={"pointsAllowed": "opp_avg_points_allowed"}))

    df = players.merge(season_avg, on="PlayerName", how="left")
    df = df.merge(team_def,left_on="opponentteamName",right_on="teamName",how="left",)

    df = df.dropna(subset=["last5_avg_points","season_avg_points","opp_avg_points_allowed","points",])

    feature_cols = ["last5_avg_points", "season_avg_points", "opp_avg_points_allowed"]
    X = df[feature_cols]
    y = df["points"]
    return X, y, team_def, season_avg



def main():
    X, y, team_def, season_avg = build_feature_table()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = mse**0.5
    r2 = r2_score(y_test, preds)

    joblib.dump(model, MODEL_DIR / "points_regressor.pkl")
    team_def.to_csv(MODEL_DIR / "team_defense_reference.csv", index=False)
    season_avg.to_csv(MODEL_DIR / "season_avg_reference.csv", index=False)

if __name__ == "__main__":
    main()
