# sentiment-analysis
predict 3 expressions positive,negative,nutral.
project directory structure
sentiment-analysis-tool/
├── data/
│   ├── raw/
│   │   └── sample_data.csv
│   ├── processed/
├── models/                      # Directory for model files
│   ├── vectorizer.pkl           # Will be created by the script
│   ├── classifier.pkl           # Will be created by the script
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── sentiment_analysis.py
│   ├── evaluation.py
├── tests/
├── requirements.txt
├── README.md
└── app.py
choose dataset according to your preference 
execute
pyhton run.py
it will generate 
vectorizer.pkl and classifier.pkl
now data is trained 
execute
python app.py
