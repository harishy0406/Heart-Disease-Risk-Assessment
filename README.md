
A web application that uses machine learning to assess the risk of heart disease based on user-provided health metrics.

## 🎯 Project Overview

This project combines machine learning with a user-friendly web interface to provide instant heart disease risk assessments. The application uses a Random Forest classifier trained on the [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) from Kaggle.

## ✨ Features

- **AI-Powered Risk Assessment**: Uses a trained Random Forest model for accurate predictions
- **Modern Web Interface**: Beautiful, responsive design with hero section and intuitive forms
- **Real-time Predictions**: Get instant risk assessments with probability scores
- **User-Friendly**: Simple form-based input for health metrics
- **Privacy-Focused**: Data is processed securely and not stored

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Flask**: Web framework for the application
- **Scikit-Learn**: Machine learning library (Random Forest)
- **Pandas & NumPy**: Data processing
- **KaggleHub**: Dataset loading
- **HTML/CSS/JavaScript**: Frontend interface

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Kaggle account (for dataset access)

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/harishy0406/Heart-Disease-Risk-Assessment.git
cd Heart-Disease-Risk-Assessment
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

Open and run the Jupyter notebook `heart_disease_model_training.ipynb`:

```bash
jupyter notebook heart_disease_model_training.ipynb
```

**Note**: You'll need to authenticate with Kaggle to download the dataset. The notebook will:
- Load the heart disease dataset from Kaggle
- Preprocess and explore the data
- Train a Random Forest classifier
- Save the model as `models/heart_disease_model.pkl`
- Save metadata as `models/model_metadata.pkl`

### 4. Run the Flask Application

After training the model, start the Flask server:

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## 📁 Project Structure

```
Heart-Disease-Risk-Assessment/
│
├── heart_disease_model_training.ipynb  # Jupyter notebook for model training
├── app.py                              # Flask application
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
│
├── models/                             # Model files (created after training)
│   ├── heart_disease_model.pkl        # Trained Random Forest model
│   └── model_metadata.pkl             # Model metadata and encoders
│
├── templates/                          # HTML templates
│   ├── index.html                     # Landing page
│   └── assessment.html                # Assessment form page
│
└── static/                             # Static files
    └── style.css                      # CSS stylesheet
```

## 🎨 Application Features

### Landing Page (`/`)
- Hero section with gradient background
- Navigation bar
- About section with feature cards
- Call-to-action button to start assessment

### Assessment Page (`/assessment`)
- Comprehensive form for health metrics:
  - Age
  - Sex
  - Chest Pain Type
  - Resting Blood Pressure
  - Cholesterol
  - Fasting Blood Sugar
  - Resting ECG
  - Maximum Heart Rate
  - Exercise-Induced Angina
  - ST Depression (Oldpeak)
  - ST Slope
- Real-time risk assessment
- Results display with probability scores

## 📊 Model Information

- **Algorithm**: Random Forest Classifier
- **Features**: Multiple health metrics including age, blood pressure, cholesterol, etc.
- **Output**: Binary classification (Heart Disease / No Heart Disease) with probability scores

## 🔧 Usage

1. **Start the Application**: Run `python app.py`
2. **Navigate to Landing Page**: Open `http://localhost:5000` in your browser
3. **Start Assessment**: Click the "Start Assessment" button
4. **Fill the Form**: Enter or select your health metrics
5. **Get Results**: Click "Assess Risk" to get your risk assessment

## ⚠️ Important Notes

- **Medical Disclaimer**: This application is for educational and informational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical concerns.

- **Model Accuracy**: The model's accuracy depends on the training data and may vary. Results are probabilistic and should be interpreted with caution.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 👤 Author

**Harish Gautham**

- GitHub: [@harishy0406](https://github.com/harishy0406)

## 🙏 Acknowledgments

- Dataset: [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) by johnsmith88 on Kaggle
- Flask community for the excellent web framework
- Scikit-learn team for the machine learning tools

---

**Built with ❤️ using Flask and Machine Learning**
