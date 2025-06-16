from flask import Flask, request, render_template
import json
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load FAQ data
with open('faq_data.json', 'r') as f:
    faq_data = json.load(f)

# Prepare questions and answers
questions = [faq['Question'] for faq in faq_data]
answers = [faq['Answer'] for faq in faq_data]

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

cleaned_questions = [clean_text(q) for q in questions]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_questions)

# Fun facts to show with every response
subject_facts = [
    "🐍 Python was named after Monty Python, not the snake!",
    "🌐 HTML stands for HyperText Markup Language.",
    "🎨 CSS lets you style HTML — think fonts, colors, layouts!",
    "🚀 Flask is a lightweight web framework in Python.",
    "⚡ JavaScript can make websites interactive.",
    "📦 ‘pip’ stands for ‘Pip Installs Packages’!",
    "🧠 Debugging is harder than writing the code itself!",
    "💡 Google, Instagram, and Spotify all use Python.",
    "🛠 VS Code is one of the most popular code editors."
]

# Get answer and fun facts
def get_answer(user_input):
    user_input_clean = clean_text(user_input)
    selected_facts = random.sample(subject_facts, 3)

    if not user_input_clean.strip():
        return random.choice(subject_facts), selected_facts

    user_vec = vectorizer.transform([user_input_clean])
    similarities = cosine_similarity(user_vec, X)

    best_match_index = similarities.argmax()
    best_score = similarities[0][best_match_index]

    if best_score > 0.2:
        best_answer = answers[best_match_index]
    else:
        best_answer = "I'm not sure. Can you try asking differently?"

    return best_answer, selected_facts

# Flask app setup
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    answer = ""
    facts = []
    if request.method == "POST":
        user_question = request.form["question"]
        answer, facts = get_answer(user_question)
    return render_template("index.html", answer=answer, facts=facts)

if __name__ == "__main__":
    app.run(debug=True)