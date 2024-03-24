from text_generation import generate_text
from text_preprocessing import preprocess_text
from tfidf_calculation import calculate_tfidf, normalize_tfidf, calculate_tfidf_sklearn

# Generate text with a sports-related prompt
generated_text = generate_text("Sports are", max_length=100)

# Preprocess generated text
preprocessed_text = preprocess_text(generated_text)

# Print preprocessed text
print("Preprocessed text:")
print(preprocessed_text)

# Calculate TF-IDF from scratch
tfidf_scratch = calculate_tfidf([preprocessed_text])
normalized_tfidf_scratch = normalize_tfidf(tfidf_scratch)

# Print TF-IDF values from scratch
print("\nTF-IDF from scratch:")
print(normalized_tfidf_scratch)

# Calculate TF-IDF using scikit-learn
sklearn_tfidf = calculate_tfidf_sklearn([preprocessed_text])

# Print TF-IDF values from scikit-learn
print("\nTF-IDF using scikit-learn:")
print(sklearn_tfidf)

