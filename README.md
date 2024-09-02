
# Quora Duplicate Questions Pair Classification

This project aims to classify pairs of questions from the Quora dataset to determine if they are duplicates. By accurately identifying duplicate questions, the goal is to reduce redundant content on the platform, improving user experience and search relevance.


## Table of Contents

- [Dependencies](#dependencies)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Text Representation](#text-representation)
- [Model Training](#model-training)
- [Results](#results)
- [Usage](#usage)
- [References](#references)

## Dependencies

The following Python libraries are required for this project:

Here is the list of tools and libraries used from the provided imports:

The following Python libraries are required for this project:

- **Core Libraries**: NumPy, Pandas, Matplotlib, Seaborn
- **Text Processing**: re, contractions, NLTK, FuzzyWuzzy, Gensim
- **Machine Learning**: Scikit-learn, XGBoost
- **Others**: Pickle

Install the necessary packages:
```bash
pip install numpy pandas matplotlib seaborn contractions fuzzywuzzy nltk gensim scikit-learn xgboost
```

## Data Preprocessing
- **Load Data:** The dataset is loaded from a CSV file.
- **Handle Missing Values:** Rows with missing values are dropped.
- **Sample Data:** A subset of 10,000 rows is used for analysis.
- **Expand Contractions:** Contractions in the text are expanded (e.g., "don't" → "do not").
- **Text Cleaning:** Convert text to lowercase, remove HTML tags, and perform tokenization.


## Feature Engineering

1. **Custom Features:** Features such as the length of questions, the number of common words, and stopwords are computed.
2. **Advanced Features:** Token-based features like common word ratios and fuzzy string matching scores are created.
3. **Normalization:** Features are normalized using MinMaxScaler.
4. **Text Representation:** A Bag-of-Words (BOW) representation is created using `CountVectorizer`.

## Model Training

The following models were trained and evaluated:

| Model                   | Accuracy  |
|-------------------------|-----------|
| Random Forest Classifier | 78%       |
| Logistic Regression      | 74%       |
| Gaussian Naive Bayes     | 62%       |
| K Nearest Neighbors      | 67%       |
| XGBoost Classifier       | 77%       |

The Random Forest Classifier was chosen due to its higher accuracy and lower Type II error.


## Results

The following models are trained and evaluated:

- `Random Forest Classifier`: Achieved a high accuracy score(**78%**).
- `Logistic Regression`, `GaussianNB`, `KNN`, and `XGBoost classifiers` are also evaluated.

## Usage

1. **Run the Code:**
   - Ensure all dependencies are installed.
   - Execute the notebook in a Jupyter environment or Google Colab.
   - The trained models and vectorizers will be saved as pickle files.

2. **Load Models and Vectorizers:**

   ```python
   import pickle

   # Load models and vectorizers
   cv = pickle.load(open('cv.pkl', 'rb'))
   rf = pickle.load(open('rf.pkl', 'rb'))
   min_scaler = pickle.load(open('scaler.pkl', 'rb'))
   ```

## References

- [Fuzzywuzzy: Fuzzy String Matching in Python](https://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/)
- [Quora Duplicate Questions Dataset](https://www.kaggle.com/competitions/quora-question-pairs)


## How to Run

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/manishKrMahto/Quora-Duplicate-Question-Pair.git
    cd Quora-Duplicate-Question-Pair
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Model**:
    - Execute the Jupyter Notebook or Python script to train and evaluate the model.

4. **Interact with the Model**:
    - Use the provided scripts to input question pairs and receive predictions on their duplicate status.
    ```bash
    streamlit run app.py 
    ````

## Future Improvements

- **Model Enhancements:** Explore deep learning models like LSTM or BERT for improved accuracy.
- **Feature Engineering:** Integrate more sophisticated text embeddings and similarity measures.
- **Deployment:** Deploy the model as a web service for real-time predictions using Flask or Streamlit.
- **Data Augmentation:** Enhance preprocessing steps to handle more diverse question types.

