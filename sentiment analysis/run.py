from src.data_preprocessing import load_data, split_data
from src.model_training import train_model

# Load and preprocess the data
data, labels = load_data()
X_train, X_test, y_train, y_test = split_data(data, labels)

# Train the model and save the vectorizer and classifier
train_model(X_train, y_train)

