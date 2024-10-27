import matplotlib.pyplot as plt

# Define the categories for the x-axis
categories = ["VEP", "R3M", "VC-1", "CLIP", "MVP"]

# Assuming you have some data corresponding to each category
data_values = [4.22, 2.11, 2.19, 1.87, 2.35]  # You should replace these values with your actual data

# Create the bar plot
plt.bar(categories, data_values, color=['orange', 'blue', 'green', 'red', 'purple'], width=0.6)

# Add labels and title
plt.ylabel('Mean reward')
plt.title('Mean rerard of South Shore')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show plot
plt.savefig('SouthShore.png')

# import matplotlib.pyplot as plt

# # Define the categories for the x-axis
# categories = ["VEP", "R3M", "VC-1", "CLIP", "MVP"]

# # Assuming you have some data corresponding to each category
# data_values = [9.6, 1.49, 1.14, 0.89, 0.95]  # You should replace these values with your actual data

# # Create the bar plot
# plt.bar(categories, data_values, color=['orange', 'blue', 'green', 'red', 'purple'], width=0.6)

# # Add labels and title
# plt.ylabel('Mean reward')
# plt.title('Mean rerard of South Shore')

# # Rotate x-axis labels for better readability
# plt.xticks(rotation=45)

# # Show plot
# plt.savefig('CMU.png')