
import json
def process_file(file_path):
    category_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            components = line.strip().split(',')

            category = components[0]
            game = components[1]
            values = list(map(float, components[2:]))

            if category not in category_dict:
                category_dict[category] = {}

            category_dict[category][game] = values

    return category_dict

file_path = 'atari.csv'  # Replace with your file path
output_path = './out_dict'
result = process_file(file_path)

# with open(output_path, 'w') as json_file:
#     json.dump(result, json_file, indent=4)
    
print(result.keys())