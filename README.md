# AI vs Human Text Classification 📚🤖

This project builds a **text classification pipeline** to distinguish between human-written and AI-generated text using a simple and interpretable **Logistic Regression model**.

> 💡 GitHub Repo: [M21-22/AI_HUMAN_POST](https://github.com/M21-22/ai_human_post)

---

## 🧠 Objective

Classify text samples as either:

- **0**: Human-written  
- **1**: AI-generated  

using a supervised machine learning approach on a dataset from Kaggle.

---

## 📁 Dataset

- **Source**: [AI vs Human Text Dataset on Kaggle](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text)
- **Fields**:
  - `text`: The text content (human or AI-generated)
  - `generated`: Label (0 for human, 1 for AI)

> ⚠️ Note: Due to license concerns, the dataset is not included. Please download it manually from the Kaggle link above and place it as `AI_human.csv` in the root folder.

---

## 🛠️ Workflow

1. **Text Cleaning**
   - Lowercasing
   - Removing punctuation, numbers, and URLs
   - Removing stopwords
   - Lemmatization using NLTK

2. **Vectorization**
   - TF-IDF (`TfidfVectorizer`) with max 5000 features

3. **Modeling**
   - `LogisticRegression` with `max_iter=1000`

4. **Evaluation**
   - Accuracy score
   - Precision, recall, F1-score

---

## 🔍 Example Output

```
Accuracy: 0.992
              precision    recall  f1-score   support

         0.0       0.99      1.00      0.99     61159
         1.0       0.99      0.99      0.99     36288

    accuracy                           0.99     97447
   macro avg       0.99      0.99      0.99     97447
weighted avg       0.99      0.99      0.99     97447
```

---

## 📦 Dependencies

- `pandas`
- `nltk`
- `scikit-learn`

Install them via:

```bash
pip install pandas nltk scikit-learn
```

Also run this once to download NLTK data:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/ai-vs-human-text-classifier.git
   cd ai-vs-human-text-classifier
   ```

2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text) and place it as `AI_human.csv`.

3. Run the main script:
   ```bash
   python main.py
   ```

---

## 📌 Notes

- The model is intentionally kept simple for interpretability and baseline benchmarking.
- You can try other models like XGBoost, SVM, or deep learning for improvement.

---

## 📄 License

This project is for **educational purposes** only.  
Check the dataset license on Kaggle before redistributing or publishing any models trained on it.

---

## 🤝 Credits

- Dataset by [Shane Gerami](https://www.kaggle.com/shanegerami)  
- Developed by [@M21-22](https://github.com/M21-22)
