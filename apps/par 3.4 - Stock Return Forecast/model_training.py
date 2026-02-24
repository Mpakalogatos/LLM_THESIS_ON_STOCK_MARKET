# ------------------ model_training.py ------------------

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------ Utilities ------------------

def time_series_split(X, y, train_ratio=0.7):
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    train_size = int(len(X) * train_ratio)
    return X.iloc[:train_size], X.iloc[train_size:], y.iloc[:train_size], y.iloc[train_size:]

def train_model(model_class, X, y):
    X_train, X_test, y_train, y_test = time_series_split(X, y)
    model = model_class()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_test, y_pred

# Model factories
def random_forest(): return RandomForestRegressor(n_estimators=100, random_state=42)
def extra_trees(): return ExtraTreesRegressor(n_estimators=300, random_state=42)
def svr_model(): return SVR(kernel='rbf')
def mlp_model(): return MLPRegressor(hidden_layer_sizes=(50,50), max_iter=1000, random_state=42)
def gradient_boosting(): return GradientBoostingRegressor(n_estimators=100, random_state=42)

# ------------------ Streamlit App ------------------

st.title("🤖 Stock Change Prediction - Model Training")

uploaded_file = st.file_uploader("Upload merged data CSV", type=["csv"])

if uploaded_file:
    merged_data = pd.read_csv(uploaded_file)
    st.subheader("📊 Loaded Data")
    st.dataframe(merged_data.head(20))

    X = merged_data[['Volume','Avg_Sentiment_Score', 'Prev_Pct_Change']]
    y = merged_data['Pct_Change']

    model_choice = st.radio(
        "Select model:",
        ["Random Forest", "Extra Trees", "SVR", "MLP", "Gradient Boosting"]
    )

    model_map = {
        "Random Forest": random_forest,
        "Extra Trees": extra_trees,
        "SVR": svr_model,
        "MLP": mlp_model,
        "Gradient Boosting": gradient_boosting
    }

    if st.button("Train Model"):
        model_class = model_map[model_choice]
        model, y_test, y_pred = train_model(model_class, X, y)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
        st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
        st.write(f"**R² Score:** {r2:.2f}")

        # Scatter plot
        fig1, ax1 = plt.subplots(figsize=(8,5))
        sns.scatterplot(x=y_test, y=y_pred, color='blue', ax=ax1)
        ax1.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label="Perfect Prediction")
        ax1.set_xlabel("Actual % Change")
        ax1.set_ylabel("Predicted % Change")
        ax1.legend()
        st.pyplot(fig1)

        # Display predicted dataframe
        pred_df = pd.DataFrame({
        "Actual_Pct_Change": y_test,
        "Predicted_Pct_Change": y_pred
        }).reset_index(drop=True)

        st.subheader("📄 Predictions vs Actual")
        st.dataframe(pred_df)