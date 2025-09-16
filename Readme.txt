# Movie Review Sentiment Analysis — Model Comparison

This project compares four machine learning algorithms (Naive Bayes, Logistic Regression, Linear SVM, Random Forest) for sentiment analysis on the NLTK movie reviews dataset.

## Project Structure

```
main.py
requirements.txt
outputs/
    model_comparison.csv
Demo.mp4
```

## Setup Instructions

1. **Clone or Download the Repository**

2. **Install Dependencies**

   Make sure you have Python 3.7+ installed. Then, install the required packages:

   ```sh
   pip install -r requirements.txt
   ```

3. **Download NLTK Data**

   The script will automatically download the `movie_reviews` corpus if not already present.

4. **Run the Script**

   Execute the main script to train models and compare their performance:

   ```sh
   python main.py
   ```

   This will:
   - Train and evaluate all four models.
   - Print a comparison table in the terminal.
   - Save the results to `outputs/model_comparison.csv`.

5. **View Results**

   - Check the terminal output for model performance.
   - Open `outputs/model_comparison.csv` for a summary table.

## Notes

- The script uses the NLTK `movie_reviews` dataset, which contains 2,000 labeled movie reviews.
- The results may vary slightly due to random train/test splits.

## Demo

See `Demo.mp4` for a demonstration of the workflow.

---