from flask import Flask, render_template, request
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import os

app = Flask(__name__)


analyzer = SentimentIntensityAnalyzer()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        
        file = request.files['file']
        if file and file.filename.endswith('.xlsx'):
            filepath = os.path.join('uploads', file.filename)
            file.save(filepath)

            
            data = pd.read_excel(filepath)
            if 'Comment' not in data.columns:
                return "Error: The uploaded file must contain a 'Comment' column."

            
            data['Sentiment'] = data['Comment'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
            data['Sentiment_Label'] = data['Sentiment'].apply(lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral'))

            
            sentiment_counts = data['Sentiment_Label'].value_counts()
            labels = sentiment_counts.index
            sizes = sentiment_counts.values
            plt.figure(figsize=(8, 6))
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
            plt.axis('equal')

            
            chart_path = os.path.join('static', 'sentiment_pie_chart.png')
            plt.savefig(chart_path)

            return render_template('results.html', table=data.to_html(classes='table table-striped', index=False), chart='sentiment_pie_chart.png')

    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)

