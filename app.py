from flask import Flask, request, render_template
import spacy
from collections import Counter
from textblob import TextBlob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Load the courses data
courses_df = pd.read_csv(r"C:\Users\Administrator\Downloads\archive (11)\alison.csv")

# Normalize the column names
courses_df.columns = courses_df.columns.str.strip()

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Predefined keyword mappings for abbreviations
keyword_mappings = {
    'ai': 'artificial intelligence',
    'ml': 'machine learning',
    'aiml': 'artificial intelligence and machine learning'
}

# Function to preprocess course descriptions for classifier training
def preprocess_data(courses_df):
    courses_df['Name Of The Course'] = courses_df['Name Of The Course'].str.lower().str.strip()
    courses_df['Description'] = courses_df['Description'].str.lower().str.strip()
    courses_df['Label'] = courses_df['Description'].apply(lambda x: 1 if 'machine learning' in x else 0)
    return courses_df

# Function to train Decision Tree classifier
def train_classifier(courses_df, max_depth=None):
    # Transform the course descriptions into a bag-of-words representation
    X = vectorizer.fit_transform(courses_df['Description'].fillna(''))
    y = courses_df['Label']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the Decision Tree Classifier with the specified max_depth
    classifier = DecisionTreeClassifier(max_depth=None)
    
    # Fit the classifier to the training data
    classifier.fit(X_train, y_train)
    
    return classifier
# Train classifier on course descriptions
courses_df = preprocess_data(courses_df)
classifier = train_classifier(courses_df)

def extract_noun_phrases(text):
    text = text.lower()
    for abbr, full_form in keyword_mappings.items():
        text = text.replace(abbr, full_form)

    doc = nlp(text)
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    noun_phrase_counts = Counter(noun_phrases)

    filtered_noun_phrases = []
    for phrase in noun_phrase_counts:
        if "certificate" not in phrase.lower():
            parts = [part.strip() for part in phrase.split('and')]
            filtered_noun_phrases.extend(parts)

    meaningful_keywords = [
        token.text for token in doc
        if token.pos_ in ["NOUN", "PROPN", "ADJ"] and "certificate" not in token.text.lower()
    ]

    named_entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "PRODUCT", "EVENT"]]

    final_keywords = list(set(filtered_noun_phrases + meaningful_keywords + named_entities))
    return final_keywords

def recommend_courses(keywords):
    recommended_courses = pd.DataFrame()

    if 'Name Of The Course' not in courses_df.columns or 'Description' not in courses_df.columns:
        return recommended_courses

    if not keywords:
        return pd.DataFrame()

    keywords = [keyword.lower().strip() for keyword in keywords if keyword.strip()]

    input_vector = vectorizer.transform([' '.join(keywords)])
    predicted = classifier.predict(input_vector)

    if predicted[0] == 1:
        recommended_courses = courses_df[courses_df['Description'].str.contains('machine learning', na=False)]
    else:
        cosine_similarities = cosine_similarity(input_vector, vectorizer.transform(courses_df['Description'].fillna(''))).flatten()
        top_indices = np.argsort(cosine_similarities)[-5:][::-1]
        recommended_courses = courses_df.iloc[top_indices]
    
    recommended_courses = recommended_courses.drop_duplicates()
    return recommended_courses[['Name Of The Course', 'Description', 'Institute', 'Link']]

def display_syntax_semantics(text):
    doc = nlp(text)
    result = "<h2>--- Syntax and Semantics ---</h2>"
    result += f"<strong>Original Text:</strong> {text}<br><br>"

    result += "<strong>Part of Speech Tagging:</strong><br>"
    for token in doc:
        result += f"{token.text} | POS: {token.pos_} | Lemma: {token.lemma_} | Dependency: {token.dep_}<br>"

    result += "<br><strong>Named Entities:</strong><br>"
    for ent in doc.ents:
        result += f"{ent.text} ({ent.label_})<br>"

    result += "<br><strong>Noun Phrases:</strong><br>"
    for chunk in doc.noun_chunks:
        result += f"{chunk.text}<br>"

    blob = TextBlob(text)
    sentiment = blob.sentiment
    polarity = sentiment.polarity
    subjectivity = sentiment.subjectivity

    sentiment_label = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"

    result += "<br><strong>Sentiment Analysis:</strong><br>"
    result += f"Polarity: {polarity:.2f} (ranges from -1 to 1)<br>"
    result += f"Subjectivity: {subjectivity:.2f} (ranges from 0 to 1)<br>"
    result += f"Sentiment: {sentiment_label}<br>"

    return result

# Flask app setup
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    recommendation_result = ""
    syntax_result = ""

    if request.method == "POST":
        topic = request.form.get("topic")

        if topic.strip():
            syntax_result = display_syntax_semantics(topic)
            keywords = extract_noun_phrases(topic)
            recommended_courses = recommend_courses(keywords)

            if not recommended_courses.empty:
                recommendation_result = "<h2>Recommended Courses:</h2>"
                for index, row in recommended_courses.iterrows():
                    recommendation_result += f"<strong>Course Name:</strong> {row['Name Of The Course']}<br>"
                    recommendation_result += f"<strong>Description:</strong> {row['Description']}<br>"
                    recommendation_result += f"<strong>Institution:</strong> {row['Institute']}<br>"
                    recommendation_result += f"<strong>Link:</strong> <a href='{row['Link']}' target='_blank'>{row['Link']}</a><br><br>"
            else:
                recommendation_result = "No courses found for the given topic."
        else:
            syntax_result = "Please enter a topic."

    return render_template("index.html", syntax_result=syntax_result, recommendation_result=recommendation_result)

if __name__ == "__main__":
    app.run(debug=True)
