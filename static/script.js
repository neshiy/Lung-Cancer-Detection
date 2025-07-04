document.addEventListener('DOMContentLoaded', () => {
    const predictBtn = document.getElementById('predict-btn');
    const imageInput = document.getElementById('image');
    const imagePreview = document.getElementById('image-preview');
    const rfResult = document.getElementById('rf-result').querySelector('span');
    const dlResult = document.getElementById('dl-result').querySelector('span');
    const errorMsg = document.getElementById('error');

    // Preview the uploaded image
    imageInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                const img = document.createElement('img');
                img.src = event.target.result;
                imagePreview.innerHTML = '';
                imagePreview.appendChild(img);
                imagePreview.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    });

        // Handle the predict button click
    predictBtn.addEventListener('click', async () => {
        // Reset previous results and errors
        rfResult.textContent = 'Waiting...'; 
        dlResult.textContent = 'Waiting...';
        errorMsg.textContent = '';

        // Collect tabular data
        const data = {};
        const featureInputs = [
            'GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
            'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING',
            'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN'
        ];

        for (const feature of featureInputs) {
            const input = document.getElementById(feature);
            if (!input.value) {
                errorMsg.textContent = `Please fill in ${feature}`;
                return;
            }
            data[feature] = input.value;
        }

        // Validate AGE
        if (data.AGE < 0 || data.AGE > 120) {
            errorMsg.textContent = 'Age must be between 0 and 120';
            return;
        }

        // Get the image as base 64
        const file = imageInput.files[0];
        if (!file) {
            errorMsg.textContent = 'Please upload a CT image';
            return;
        }

        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = async () => {
            data.image = reader.result;

        // Send the data to the API
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                });

                const result = await response.json();

                if (response.ok) {
                    rfResult.textContent = result.rf_prediction;
                    dlResult.textContent = result.dl_prediction;
                } else {
                    errorMsg.textContent = result.error || 'An error occurred';
                }
            } catch (err) {
                errorMsg.textContent = 'Failed to connect to the server';
            }
        };
    });
});