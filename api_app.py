from flask import Flask, request, jsonify
import numpy as np
import cv2
import joblib
import torch
import torch.nn as nn
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Define the PyTorch model (same as during training)
class LungCancerModel(nn.Module):
    def __init__(self, tabular_input_dim):
        super(LungCancerModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc_image = nn.Linear(64 * 32 * 32, 128)
        self.relu = nn.ReLU()
        self.fc_tabular1 = nn.Linear(tabular_input_dim, 64)
        self.fc_tabular2 = nn.Linear(64, 32)
        self.fc_combined1 = nn.Linear(128 + 32, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc_combined2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, image, tabular):
        x1 = self.pool1(self.relu(self.conv1(image)))
        x1 = self.pool2(self.relu(self.conv2(x1)))
        x1 = self.flatten(x1)
        x1 = self.relu(self.fc_image(x1))
        x2 = self.relu(self.fc_tabular1(tabular))
        x2 = self.relu(self.fc_tabular2(x2))
        x = torch.cat((x1, x2), dim=1)
        x = self.relu(self.fc_combined1(x))
        x = self.dropout(x)
        x = self.fc_combined2(x)
        x = self.sigmoid(x)
        return x

# Load the models and preprocessors
rf_model = joblib.load('lung_cancer_rf_model.pkl')
le_gender = joblib.load('le_gender.pkl')
le_target = joblib.load('le_target.pkl')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('selected_features.pkl')

# Load the PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tabular_input_dim = len(selected_features)
dl_model = LungCancerModel(tabular_input_dim=tabular_input_dim).to(device)
dl_model.load_state_dict(torch.load('lung_cancer_dl_model.pth', map_location=device))
dl_model.eval()

# Define the feature columns (excluding RISK_SCORE, which we'll compute)
feature_columns = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
                   'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING',
                   'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        

        # Extract tabular data
        tabular_data = {}
        for feature in feature_columns:
            if feature in data:
                if feature == 'GENDER':
                    tabular_data[feature] = le_gender.transform([data[feature].upper()])[0]
                elif feature == 'AGE':
                    tabular_data[feature] = float(data[feature])
                else:
                    tabular_data[feature] = int(data[feature])
            else:
                return jsonify({'error': f'Missing feature: {feature}'}), 400

        # Create RISK_SCORE
        risk_factors = ['SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE',
                        'FATIGUE ', 'ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING',
                        'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']
        tabular_data['RISK_SCORE'] = sum(tabular_data[f] for f in risk_factors)

        # Scale AGE
        tabular_data['AGE'] = scaler.transform([[tabular_data['AGE']]])[0][0]

        # Prepare tabular input for prediction
        tabular_input = np.array([[tabular_data[f] for f in feature_columns + ['RISK_SCORE']]])
        tabular_input_selected = tabular_input[:, [feature_columns.index(f) if f in feature_columns else -1 for f in selected_features]]
        print("Tabular input selected shape:", tabular_input_selected.shape)  # Should be (1, 11)

        # Process the uploaded image (base64 encoded)
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # Decode the base64 image
        try:
            image_data = base64.b64decode(data['image'].split(',')[1])
            image = Image.open(BytesIO(image_data)).convert('L')  # Convert to grayscale
        except Exception as e:
            return jsonify({'error': f'Failed to process image: {str(e)}'}), 400

        image = np.array(image)  # Shape: (height, width)
        image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)  # Shape: (128, 128)
        image = image.astype("float32") / 255.0  # Normalize to [0, 1]

        # Reshape the image to (1, 1, 128, 128) for PyTorch (batch_size, channels, height, width)
        image = image[np.newaxis, np.newaxis, :, :]  # Shape: (1, 1, 128, 128)

        # Convert to PyTorch tensor
        image_tensor = torch.tensor(image, dtype=torch.float32).to(device)

        # Predict using the Random Forest model (tabular only)
        rf_pred = le_target.inverse_transform(rf_model.predict(tabular_input_selected))[0]

        # Predict using the PyTorch deep learning model (tabular + image)
        tabular_tensor = torch.tensor(tabular_input_selected, dtype=torch.float32).to(device)
        with torch.no_grad():
            dl_pred_proba = dl_model(image_tensor, tabular_tensor).cpu().numpy().flatten()[0]
        dl_pred = 'YES' if dl_pred_proba > 0.5 else 'NO'

        # Return the predictions as JSON
        return jsonify({
            'rf_prediction': rf_pred,
            'dl_prediction': dl_pred
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
    