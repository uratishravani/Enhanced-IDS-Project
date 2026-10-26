from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from flask_cors import CORS
import numpy as np
import joblib
import os
import shap

app = Flask(__name__)
CORS(app)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "final_cnn_bigru_attention_alo.keras"
)

ENCODER_PATH = os.path.join(
    BASE_DIR, "data", "encoder.pkl"
)

MEAN_PATH = os.path.join(
    BASE_DIR, "data", "train_mean.npy"
)

RANGE_PATH = os.path.join(
    BASE_DIR, "data", "train_range.npy"
)

MVO_PATH = os.path.join(
    BASE_DIR, "data", "selected_feature_indices.npy"
)

# ============================================================
# LOAD MODEL AND PREPROCESSING FILES
# ============================================================

print("Loading model...")

model = load_model(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

train_mean = np.load(MEAN_PATH)
train_range = np.load(RANGE_PATH)

selected_indices = np.load(MVO_PATH)

print("Model loaded successfully!")
print("Model input shape:", model.input_shape)
print("Encoder features:", len(encoder.get_feature_names_out()))
print("Mean shape:", train_mean.shape)
print("Range shape:", train_range.shape)
print("MVO selected features:", len(selected_indices))

# ============================================================
# NUMERICAL FEATURES
# ============================================================

numerical_features = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate"
]

# ============================================================
# FEATURE NAMES
# ============================================================

encoder_feature_names = list(
    encoder.get_feature_names_out(
        ["protocol_type", "service", "flag"]
    )
)

all_feature_names = numerical_features + encoder_feature_names

selected_feature_names = [
    all_feature_names[int(index)]
    for index in selected_indices
]

print("Total feature names:", len(all_feature_names))
print("Selected feature names:", len(selected_feature_names))

# ============================================================
# PREPROCESSING FUNCTION
# ============================================================

def preprocess_input(data):

    # Numerical values
    numerical_values = []

    for feature in numerical_features:

        if feature not in data:
            raise ValueError(
                f"Missing feature: {feature}"
            )

        numerical_values.append(
            float(data[feature])
        )

    # Categorical values
    protocol_type = data.get("protocol_type")
    service = data.get("service")
    flag = data.get("flag")

    if protocol_type is None:
        raise ValueError(
            "Missing feature: protocol_type"
        )

    if service is None:
        raise ValueError(
            "Missing feature: service"
        )

    if flag is None:
        raise ValueError(
            "Missing feature: flag"
        )

    # Numerical array
    numerical_array = np.array(
        numerical_values,
        dtype=np.float32
    ).reshape(1, -1)

    # One-hot encoding
    categorical_array = encoder.transform(
        [[protocol_type, service, flag]]
    )

    # Combine numerical + categorical
    combined_features = np.hstack(
        [numerical_array, categorical_array]
    )

    if combined_features.shape[1] != 122:

        raise ValueError(
            f"Expected 122 features, "
            f"got {combined_features.shape[1]}"
        )

    # Mean normalization
    normalized_features = (
        combined_features - train_mean
    ) / train_range

    # MVO feature selection
    selected_features = normalized_features[
        :, selected_indices
    ]

    # CNN input shape
    model_input = selected_features.reshape(
        1, 33, 1
    )

    return model_input


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return "Enhanced IDS Backend is running!"


# ============================================================
# PREDICTION API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        model_input = preprocess_input(data)

        prediction = model.predict(
            model_input,
            verbose=0
        )[0][0]

        result = (
            "Attack"
            if prediction >= 0.5
            else "Normal"
        )

        return jsonify({

            "prediction": float(prediction),

            "result": result,

            "total_features": 122,

            "selected_features": 33

        })

    except Exception as e:

        return jsonify({

            "error": str(e)

        }), 500


# ============================================================
# SHAP EXPLANATION FUNCTION
# ============================================================

def model_predict_for_shap(X):

    """
    SHAP gives input as:

        (samples, 33)

    But our CNN-BiGRU model expects:

        (samples, 33, 1)

    So we reshape before prediction.
    """

    X = np.asarray(X)

    X = X.reshape(
        X.shape[0],
        33,
        1
    )

    predictions = model.predict(
        X,
        verbose=0
    )

    return predictions.reshape(-1)


# ============================================================
# SHAP API
# ============================================================

@app.route("/explain", methods=["POST"])
def explain():

    try:

        data = request.get_json()

        # ----------------------------------------------------
        # Preprocess input
        # ----------------------------------------------------

        model_input = preprocess_input(data)

        # Convert from
        # (1, 33, 1)
        #
        # to
        # (1, 33)

        shap_input = model_input.reshape(
            1,
            33
        )

        # ----------------------------------------------------
        # SHAP background
        # ----------------------------------------------------

        background = np.zeros(
            (1, 33),
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Create KernelExplainer
        # ----------------------------------------------------

        explainer = shap.KernelExplainer(
            model_predict_for_shap,
            background
        )

        # ----------------------------------------------------
        # Calculate SHAP values
        # ----------------------------------------------------

        shap_values = explainer.shap_values(
            shap_input,
            nsamples=50
        )

        # SHAP version can return different structures
        if isinstance(shap_values, list):

            shap_values = shap_values[0]

        shap_values = np.asarray(
            shap_values
        )

        # Make sure we get one-dimensional array
        shap_values = shap_values.reshape(-1)

        # ----------------------------------------------------
        # Create feature explanations
        # ----------------------------------------------------

        explanations = []

        for i in range(33):

            explanations.append({

                "feature":
                    selected_feature_names[i],

                "shap_value":
                    float(shap_values[i]),

                "feature_value":
                    float(shap_input[0][i])

            })

        # ----------------------------------------------------
        # Sort by absolute SHAP importance
        # ----------------------------------------------------

        explanations.sort(
            key=lambda x:
                abs(x["shap_value"]),
            reverse=True
        )

        # ----------------------------------------------------
        # Top 10 features
        # ----------------------------------------------------

        top_features = explanations[:10]

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            model_input,
            verbose=0
        )[0][0]

        result = (
            "Attack"
            if prediction >= 0.5
            else "Normal"
        )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return jsonify({

            "prediction":
                float(prediction),

            "result":
                result,

            "top_features":
                top_features

        })

    except Exception as e:

        print("SHAP ERROR:", e)

        return jsonify({

            "error": str(e)

        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000
    )