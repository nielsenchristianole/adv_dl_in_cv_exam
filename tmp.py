import pandas as pd

def load_and_change_names(filename):
    # Load the CSV file
    df = pd.read_csv(filename)

    # Define a function to change the names
    def change_name(name):
        # Remove the .jpg extension
        return name.replace('.jpg', '')

    # Apply the function to the 'path' column
    df['name'] = df['name'].apply(change_name)

    return df

# Use the function
df = load_and_change_names('data/elo_annotations/calle2.csv')

# save the dataframe
df.to_csv('data/elo_annotations/calle3.csv', index=False)

