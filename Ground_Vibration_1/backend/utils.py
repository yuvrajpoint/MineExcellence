import numpy as np
import pandas as pd
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Use a headless backend for non-GUI environments

import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import linregress
from sklearn.metrics import r2_score
import joblib
import math

model = joblib.load("/Users/yuvraj/Downloads/MineExcellence/ppv_rf_model.pkl")
df = pd.read_csv("/Users/yuvraj/Downloads/MineExcellence/merged_cleaned_4.csv")

features = [
    'Hole dia. [mm]', 'Hole depth [m]', 'No. of holes', 'Avg. Burden [m]', 'Avg.Spacing [m]', 'Avg. top stemming length [m]', 'Total charge [kg]', 'Max.charge delay [kg]', 'distance', 'Pit'
]

def formula_ppv(user_input: dict):
    d = user_input['distance']
    q = user_input['Max.charge delay [kg]']
    if user_input['Pit'] == 1:
        ppv = 696 * ((d / (math.sqrt(q))) ** (-1.21))
        return ppv
    else:
        ppv = 3576 * ((d / (math.sqrt(q))) ** (-1.59))
        return ppv

def predict_ppv(user_input):
    if isinstance(user_input, dict):
        input_df = pd.DataFrame([user_input])[features]  # ensure correct column order
    elif isinstance(user_input, pd.DataFrame):
        input_df = user_input[features]  # select only needed features
    else:
        raise ValueError("user_input must be a dict or DataFrame")

    prediction = model.predict(input_df)
    return prediction[0]


# Generate predictions for a range of distances
def generate_predictions(base_params):
    distances = list(range(100, 451, 10))
    ml_preds = []
    formula_preds = []
    
    for d in distances:
        user_input_copy = base_params.copy()
        user_input_copy['distance'] = d
        formula_preds.append(formula_ppv(user_input_copy))
        ml_pred = predict_ppv(user_input_copy)
        # print(ml_pred)
        ml_preds.append(ml_pred)
        
    # Clip out initial rising part — drop values below 140m
    clipped_distances = [d for d in distances if d >= 170]
    start_idx = distances.index(clipped_distances[0])
    
    return (
        clipped_distances,
        ml_preds[start_idx:],      # trim predictions
        formula_preds[start_idx:], # trim formula
    )

# Extract actual data points from dataset for a given pit
def get_actual_points(blast):
    df_filtered = df[df['blast'] == blast]
    return df_filtered['distance'][1:], df_filtered['ppv'][1:]

def graph(base_params, bno):
    ppv = predict_ppv(base_params)
    bno = "B" + str(bno)
    # Generate predicted and actual data
    distances, ml_preds, formula_preds = generate_predictions(base_params)
    print('we are after prediction', ppv)
    actual_d, actual_ppv = get_actual_points(bno)

    print("we are here")
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))   

    # Plot predictions and actuals
    ax.plot(distances, ml_preds, label="ML Model Prediction (Smoothed)", color='green')
    ax.plot(distances, formula_preds, label="Empirical Formula (Smoothed)", color='orange')
    ax.scatter(actual_d, actual_ppv, label="Actual Measured PPV", color='blue', s=40, marker='o')

    # Add labels and title
    ax.set_title(f"PPV Prediction Comparison for Pit {bno}")
    ax.set_xlabel("Distance from Blast (m)")
    ax.set_ylabel("PPV")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    img = fig_to_base64(fig)

    # Return base64-encoded image
    return {'ppv': ppv, 'graph': img}

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def fit_usb_model(distances, charges, ppv_preds):
    # Calculate scaled distance: D / sqrt(Q)
    scaled_distance = distances / np.sqrt(charges)
    
    # Take logs
    log_scaled_distance = np.log(scaled_distance)
    log_ppv = np.log(ppv_preds)
    
    # Linear regression: log(PPV) = log(k) - b * log(SD)
    slope, intercept, r_value, p_value, std_err = linregress(log_scaled_distance, log_ppv)
    
    b = -slope
    k = np.exp(intercept)
    
    return k, b, r_value**2

def generate_log_log_graphs(base_params):
    # Distance range around user distance
    user_distance = base_params['distance']
    distances = np.linspace(max(user_distance - 200, 1), user_distance + 200, 50)
    ppv_preds = []

    for d in distances:
        user_input_copy = base_params.copy()
        user_input_copy['distance'] = d
        ml_pred = predict_ppv(user_input_copy)
        # print(ml_pred)
        ppv_preds.append(ml_pred)

    # PPV vs Distance
    fig1, ax1 = plt.subplots()
    ax1.plot(distances, ppv_preds, color='orange')
    ax1.set_xlabel("Distance (m)")
    ax1.set_ylabel("Predicted PPV")
    ax1.set_title("PPV vs Distance")
    img1 = fig_to_base64(fig1)

    # Log-Log PPV vs Scaled Distance
    scaled_distance = distances / np.sqrt(base_params['Total charge [kg]'])

    fig2, ax2 = plt.subplots()

    # Plot the points
    ax2.plot(scaled_distance, ppv_preds, marker='o', color='green', linestyle='')

    # Log-log scale
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    # Labels and Title
    ax2.set_xlabel("Scaled Distance (m/√kg)")
    ax2.set_ylabel("PPV (mm/s)")
    ax2.set_title("Actual Attenuation with ML Model")

    # Add grid lines (both major and minor)
    ax2.grid(True, which='both', linestyle='-', color='gray', linewidth=0.5, alpha=0.8)

    # Enable minor ticks for dense grid
    ax2.minorticks_on()

    # Optional: cleaner look (remove top and right borders)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Convert to base64 for frontend
    img2 = fig_to_base64(fig2)


    return {"ppv_dist": img1, "log_log": img2}

user = {
    'Hole dia. [mm]': 115,
    'Hole depth [m]': 8,
    'No. of holes': 12,
    'Avg. Burden [m]': 4,
    'Avg.Spacing [m]': 6,
    'Avg. top stemming length [m]': 2.5,
    'Avg. charge/hole [kg]': 30,
    'Total charge [kg]': 200,
    'Max.charge delay [kg]': 60,
    'distance': 200,
  }


test = {"Hole dia. [mm]": 115, "Hole depth [m]": 8, "No. of holes": 12, "Avg. Burden [m]": 4, "Avg.Spacing [m]": 6, "Avg. top stemming length [m]": 2.5, "Avg. charge/hole [kg]": 30, "Total charge [kg]": 200, "Max.charge delay [kg]": 60, "distance": 200, "Pit": 1}

# graph_data = graph(test, 12)

print(generate_log_log_graphs(test))