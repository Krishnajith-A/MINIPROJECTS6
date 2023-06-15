from flask import Flask,render_template,request
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import plotly.graph_objects as go
from textblob import TextBlob

app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def io():
    return render_template('io.html')
@app.route('/about',methods=['GET','POST'])
def about():
    return render_template('about.html')
@app.route('/howto',methods=['GET','POST'])
def howto():
    return render_template('howto.html')

@app.route('/data',methods=['GET','POST'])
def data():
    # Step 1: Load the customer reviews data from CSV
    if request.method=='POST':
        file=request.form['upload-file']
        data = pd.read_csv(file)

    # Step 2: Preprocessing the Data
        def preprocess_text(text):
            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()
    
            words = word_tokenize(text.lower())
            words = [word for word in words if word.isalpha() and word not in stop_words]
            words = [lemmatizer.lemmatize(word) for word in words]
    
            return ' '.join(words)

        # Preprocess the reviews
        preprocessed_reviews = [preprocess_text(review) for review in data['review']]

        # Step 3: Perform Sentiment Analysis
        sentiments = []
        for review in preprocessed_reviews:
            blob = TextBlob(review)
            sentiment = blob.sentiment.polarity
            sentiments.append(sentiment)

        # Step 4: Extract Common Problems and Suggestions
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(preprocessed_reviews)

        # Apply dimensionality reduction for better interpretability
        svd = TruncatedSVD(n_components=5)
        lsa_matrix = svd.fit_transform(tfidf_matrix)

        # Find the most important words (features) contributing to each dimension
        feature_names = vectorizer.get_feature_names_out()
        dimension_keywords = []
        for i, component in enumerate(svd.components_):
            top_keywords = [feature_names[idx] for idx in component.argsort()[:-6:-1]]
            dimension_keywords.append(top_keywords)

        # Generate summary and suggestions
        summary = "Common problems identified in customer reviews:\n"
        suggestions = "Suggestions to improve the product:\n"
        for i, keywords in enumerate(dimension_keywords):
            summary += f"Problem {i+1}: "
            summary += ", ".join(keywords)
            summary += "\n"
    
            suggestions += f"Improve {keywords[0]} by addressing {', '.join(keywords[1:])}\n"

        # Step 5: Calculate Sentiment Percentage
        positive_percentage = (sum(sentiment > 0 for sentiment in sentiments) / len(sentiments)) * 100
        neutral_percentage = (sum(sentiment == 0 for sentiment in sentiments) / len(sentiments)) * 100
        negative_percentage = (sum(sentiment < 0 for sentiment in sentiments) / len(sentiments)) * 100

        # Step 6: Plot Pie Chart using Plotly
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [positive_percentage, neutral_percentage, negative_percentage]
        colors = ['#00ff00', '#999999', '#ff0000']

        fig = go.Figure(data=[go.Pie(labels=labels, values=sizes, textinfo='label+percent', marker=dict(colors=colors))])
        fig.update_layout(title='Sentiment Analysis', showlegend=False)
    

        # Print summary, suggestions, and sentiment percentages
        #print(summary,suggestions,"Sentiment Percentages:",f"Positive: {positive_percentage:.2f}%",f"Neutral: {neutral_percentage:.2f}%",f"Negative: {negative_percentage:.2f}%")

        return render_template("main.html",summary=summary,suggestions=suggestions, t1="Sentiment Percentages:",positive=f"Positive: {positive_percentage:.2f}%",negative=f"Negative: {negative_percentage:.2f}%",neutral=f"Neutral: {neutral_percentage:.2f}%",fig=fig.show())
if __name__=="__main__":
    app.run(debug=True)