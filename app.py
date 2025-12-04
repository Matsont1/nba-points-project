# app.py
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import streamlit as st
import joblib

MODEL_DIR = Path("models")
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "predictions_log.csv"

@st.cache_data
def load_data():
    players = pd.read_csv("PlayerStatistics.csv", low_memory=False)
    games = pd.read_csv("Games(1).csv", low_memory=False)
    players["gameDateTimeEst"] = pd.to_datetime(players["gameDateTimeEst"], errors="coerce")
    games["gameDateTimeEst"] = pd.to_datetime(games["gameDateTimeEst"], errors="coerce")
    players["PlayerName"] = players["firstName"] + " " + players["lastName"]
    players = players.sort_values(["PlayerName", "gameDateTimeEst"])
    season_avg = (players.groupby("PlayerName", as_index=False)["points"].mean().rename(columns={"points": "season_avg_points"}))
    long_rows = []
    for _, row in games.iterrows():
        long_rows.append({"teamName": row["hometeamName"], "pointsAllowed": row["awayScore"]})
        long_rows.append({"teamName": row["awayteamName"], "pointsAllowed": row["homeScore"]})
    team_def = pd.DataFrame(long_rows)
    team_def = (team_def.groupby("teamName", as_index=False)["pointsAllowed"].mean().rename(columns={"pointsAllowed": "opp_avg_points_allowed"}))
    return players, season_avg, team_def


@st.cache_resource
def load_model():
    model_path = MODEL_DIR / "points_regressor.pkl"
    return joblib.load(model_path)


def append_log(row_dict):
    """Append an entry."""
    LOG_DIR.mkdir(exist_ok=True)
    df_new = pd.DataFrame([row_dict])
    if LOG_FILE.exists():
        df_new.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df_new.to_csv(LOG_FILE, mode="w", header=True, index=False)


def load_log():
    if LOG_FILE.exists():
        return pd.read_csv(LOG_FILE)
    return None


def main():
    st.set_page_config(page_title="NBA Player Points Predictor", layout="wide")
    st.title("NBA Player Points Predictor")

    players, season_avg, team_def = load_data()
    model = load_model()

    tab_pred, tab_monitor = st.tabs(["Prediction", "Monitoring"])

    with tab_pred:
        st.subheader("Make a Prediction")

        col_controls1, col_controls2 = st.columns(2)

        with col_controls1:
            all_players = sorted(players["PlayerName"].unique())
            selected_player = st.selectbox("Player", all_players)

        player_games = players[players["PlayerName"] == selected_player].copy()
        player_games = player_games.sort_values("gameDateTimeEst")

        with col_controls2:
            opponent_options = sorted(player_games["opponentteamName"].dropna().unique())
            selected_opponent = st.selectbox("Opponent team", opponent_options)

        last5 = player_games.tail(5)
        last5_avg = float(last5["points"].mean()) if not last5.empty else np.nan

        # Season avgerage
        srow = season_avg[season_avg["PlayerName"] == selected_player]
        season_avg_points = (float(srow["season_avg_points"].iloc[0]) if not srow.empty else np.nan)

        # Opponent avg pts allowed
        orow = team_def[team_def["teamName"] == selected_opponent]
        opp_avg_allowed = (float(orow["opp_avg_points_allowed"].iloc[0]) if not orow.empty else np.nan)

        # Prediction
        if np.isnan(last5_avg) or np.isnan(season_avg_points) or np.isnan(opp_avg_allowed):
            predicted_points = np.nan
        else:
            features = np.array([[last5_avg, season_avg_points, opp_avg_allowed]])
            predicted_points = float(model.predict(features)[0])

        top_col1, top_col2, top_col3, top_col4 = st.columns(4)

        with top_col1:
            st.metric("Last 5 Games Avg",f"{last5_avg:.2f}" if not np.isnan(last5_avg) else "N/A",)

        with top_col2:
            st.metric("Season Avg Points",f"{season_avg_points:.2f}" if not np.isnan(season_avg_points) else "N/A",)

        with top_col3:
            st.metric("Opponent Avg Points Allowed",f"{opp_avg_allowed:.2f}" if not np.isnan(opp_avg_allowed) else "N/A",)

        with top_col4:
            st.metric("Predicted Points (Simple)",f"{predicted_points:.2f}" if not np.isnan(predicted_points) else "N/A",)

        st.markdown("---")

        # Last 5 games
        st.subheader(f"Last 5 Games for {selected_player}")
        if last5.empty:
            st.write("No Entries")
        else:
            chart_data = last5[["gameDateTimeEst", "points"]].copy()
            chart_data = chart_data.rename(columns={"gameDateTimeEst": "Game Date", "points": "Points"})
            chart_data = chart_data.set_index("Game Date")
            st.line_chart(chart_data)

        st.markdown("### Log this prediction")

        if np.isnan(predicted_points):
            st.info("Need valid features")
        else:
            if st.button("Log prediction"):
                row = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "player": selected_player,
                    "opponent": selected_opponent,
                    "last5_avg_points": last5_avg,
                    "season_avg_points": season_avg_points,
                    "opp_avg_allowed": opp_avg_allowed,
                    "predicted_points": predicted_points,
                    "actual_points": np.nan,
                }
                append_log(row)
                st.success("Prediction logged for monitoring.")

    # Monitoring tab
    with tab_monitor:
        st.subheader("Model Monitoring Dashboard")
    
        log_df = load_log()
        if log_df is None or log_df.empty:
            st.info("No predictions have been logged.")
            st.stop()
    
        # Show recent predictions
        st.markdown("#### Logged Prediction History")
        st.dataframe(log_df.sort_values("timestamp", ascending=False).reset_index(drop=True),use_container_width=True)
    
        st.markdown("---")
        st.subheader("Update Actual Points")
    
        # Dropdown
        log_df = log_df.sort_values("timestamp", ascending=False).reset_index(drop=True)
        options = [f"{i}: {row.timestamp} — {row.player} vs {row.opponent} "f"(Pred {row.predicted_points:.1f}, Actual {row.actual_points})"for i, row in log_df.iterrows()]
        selected_idx = st.selectbox("Select a logged prediction to update actual:",options,index=0)
        chosen_idx = int(selected_idx.split(":")[0])
    
        # Enter actual value
        new_actual = st.number_input("Actual points scored:",min_value=0.0, max_value=200.0, step=1.0)
    
        if st.button("Save Actual Value"):
            log_df.loc[chosen_idx, "actual_points"] = new_actual
            LOG_DIR.mkdir(exist_ok=True)
            log_df.to_csv(LOG_FILE, index=False)
            st.success("Saved!")
    
        st.markdown("---")
    
        # Error metrics
        valid = log_df.dropna(subset=["actual_points"]).copy()
        if valid.empty:
            st.info("No error metrics")
        else:
            valid["error"] = valid["predicted_points"] - valid["actual_points"]
            valid["abs_error"] = valid["error"].abs()
            valid["squared_error"] = valid["error"] ** 2
    
            mae = valid["abs_error"].mean()
            rmse = (valid["squared_error"].mean()) ** 0.5
    
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                st.metric("MAE (mean abs error)", f"{mae:.2f}")
            with mcol2:
                st.metric("RMSE", f"{rmse:.2f}")


if __name__ == "__main__":
    main()
