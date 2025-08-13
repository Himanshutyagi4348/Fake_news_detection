📰 Fake News Detection using Machine Learning
This project is a simple yet effective Fake News Detection System built with Python and scikit-learn. It uses a Logistic Regression classifier trained on a labeled dataset of real and fake news articles.

📂 Dataset
The project uses the Fake and Real News Dataset from Kaggle, which consists of:

Fake.csv – Fake news articles

True.csv – Real news articles

Each file includes:

title: Title of the news article

text: Full news article text

subject: Category of the article

date: Publication date

✅ Features
Combines real and fake news into one dataset

Converts news text into numerical format using TF-IDF vectorization

Trains a Logistic Regression model

Predicts whether a given news article is real or fake

Prints model evaluation metrics like accuracy and confusion matrix

Provides a function for custom predictions

🛠️ Installation
Clone the repository or download the files:
git clone https://github.com/Himanshutyagi4348/Fake_news_detection/new/main?filename=README.md.git
cd Genap\fale_news

Install the required Python libraries:

bash
Copy
Edit
pip install pandas scikit-learn
Make sure Fake.csv and True.csv are in the project directory.

🚀 Usage
Run the script:

bash
Copy
Edit
python fake_news_detector.py
Example output:

yaml
Copy
Edit
Accuracy: 0.98

📊 Classification Report:
              precision    recall  f1-score   support

           0       0.98      0.99      0.98      1067
           1       0.99      0.97      0.98      1032

    accuracy                           0.98      2099
   macro avg       0.98      0.98      0.98      2099
weighted avg       0.98      0.98      0.98      2099


🧩 Confusion Matrix:
[[1053   14]
 [  30 1002]]

📝 Sample Text: NASA is planning a new moon mission in 2025.
🔍 Prediction: 🟢 Real News
🧠 Model Details
Vectorizer: TF-IDF with English stopwords and max document frequency = 0.7

Classifier: Logistic Regression (default hyperparameters)

Split Ratio: 80% train / 20% test

Project Structure
graphql
Copy
Edit
fake-news-detector/
│
├── Fake.csv
├── True.csv
├── fake_news_detector.py
└── README.md
Example Prediction Function
You can reuse this function in any app or UI:

python
Copy
Edit
def predict_news(news_text):
    vect = vectorizer.transform([news_text])
    prediction = model.predict(vect)[0]
    return "Real News" if prediction == 1 else "Fake News"
Future Improvements
Build a web interface using Flask or Streamlit

Add more ML models like Naive Bayes or Random Forest

Improve preprocessing using NLP libraries like spaCy or NLTK

Save model with joblib for reuse

📎 License
This project is for educational purposes. If you use the dataset or code, please credit the original Kaggle dataset authors.
