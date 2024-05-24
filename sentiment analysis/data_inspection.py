import pandas as pd

def main():
    # Load the dataset
    data = pd.read_csv('data/raw/sample_data.csv')
    print(data['sentiment'].value_counts())

if __name__ == "__main__":
    main()
