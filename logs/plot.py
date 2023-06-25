import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
def plot_csv_data(csv_file,out_file):
    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(csv_file)

    # Group the data by the 'tag' column
    grouped_data = data.groupby(['tag','file'])

    # Create a new figure and axis for the plot
    fig, ax = plt.subplots(figsize=(50, 20))

    # Iterate through the groups and plot the data
    for tag, group in grouped_data:
        ax.plot(group['step'], group['value'].interpolate(method='cubic'), label=tag)

    # Add labels and a legend
    ax.set_xlabel('Step')
    ax.set_ylabel('Value')
    ax.legend()

    # Show the plot
    plt.savefig(out_file,dpi=500)

if __name__ == '__main__':
    # Replace 'path/to/your/csv_file.csv' with the path to your CSV file


    # print the list of CSV files
    plot_csv_data('./out/res.csv', 'res.png')

    # csv_file = 'final.csv'
    # plot_csv_data(csv_file,'data2.png')