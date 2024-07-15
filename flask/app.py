from flask import Flask, render_template, request, send_file
import pandas as pd
import folium
from geopy.distance import distance
from itertools import permutations
import time

app = Flask(__name__)

# Function to read a specified column from the uploaded Excel file
def read_excel_column(file_path, column_name):
    try:
        df = pd.read_excel(file_path)
        if column_name in df.columns:
            return df[column_name]
        else:
            raise ValueError(f"Column '{column_name}' not found in the Excel file.")
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        return None

# Function to process the column data and count persons
def process_data(df):
    try:
        # Check if 'Text' and 'Persons' columns exist
        if 'Text' not in df.columns or 'Persons' not in df.columns:
            raise KeyError("'Text' or 'Persons' columns not found in the DataFrame.")

        # Split 'Text' column into 'Depart' and 'Destination'
        df[['Depart', 'Destination']] = df['Text'].str.split(' to ', expand=True)

        # Handle NaN values in 'Destination' column
        df = df.dropna(subset=['Destination'])

        # Convert 'Persons' column to string to handle potential NaNs
        df['Persons'] = df['Persons'].astype(str)

        # Split the 'Persons' column and explode the dataframe to have one person per row
        df_exploded = df.assign(Persons=df['Persons'].str.split(', ')).explode('Persons')

        # Group by 'Depart' and concatenate 'Destination'
        grouped_dest = df.groupby('Depart')['Destination'].apply(lambda x: ', '.join(set(x.dropna()))).reset_index()

        # Group by 'Depart' and aggregate unique 'Persons' into a comma-separated string
        grouped_persons = df_exploded.groupby('Depart')['Persons'].apply(lambda x: ', '.join(sorted(set(x)))).reset_index()

        # Count unique persons from each 'Depart'
        person_counts = df_exploded.groupby('Depart')['Persons'].nunique().reset_index()
        person_counts.columns = ['Depart', 'Unique Person Count']

        # Assign types of cars based on the count of unique persons
        car_types = assign_car_types(person_counts)

        # Merge the grouped dataframes on 'Depart'
        grouped = pd.merge(grouped_dest, grouped_persons, on='Depart')

        return grouped, person_counts, car_types

    except KeyError as ke:
        print(f"KeyError: {ke}. Check if 'Text' and 'Persons' columns exist in the DataFrame.")
        return None, None, None
    except Exception as e:
        print(f"Error processing data: {e}")
        return None, None, None

# Function to assign types of cars based on the count of unique persons
def assign_car_types(person_counts):
    car_types = {}
    for index, row in person_counts.iterrows():
        depart = row['Depart']
        count = row['Unique Person Count']
        
        if count <= 2:
            car_types[depart] = 'Compact Car'
        elif count <= 4:
            car_types[depart] = 'Sedan'
        elif count <= 8:
            car_types[depart] = 'Minibus'
        elif count <= 12:
            car_types[depart] = 'Grand Minibus'
        else:
            car_types[depart] = 'SUV'
    
    return car_types
# Example coordinates (replace with actual coordinates)
coordinates = {
    'Haupteingang DRX Siliana':(36.0874918468941, 9.36661621046865),
    'from Hotel Mövenpick':(35.84275671442201, 10.627233236700128),
    'Haupteingang DRX METS':(35.78793688812981, 10.665982591922752),
    'from Haupteingang DRX METS':(35.7879828233093, 10.666761214992453),
    'Hotel Mövenpick': (35.84275671442201, 10.627233236700128),   #
    'Nabeul': (36.453195458024766, 10.73319058444151),                
    'Haupteingang DRX METS': (48.1351, 11.5820),  # Example coordinates for Munich, Germany
    'KA': (48.7758, 9.1829),                 # Example coordinates for Stuttgart, Germany
    'airport TUN': (36.8519, 10.2260),       # Example coordinates for Tunis, Tunisia
    'DE': (50.1109, 8.6821)                  # Example coordinates for Frankfurt, Germany
}

# Function to save the processed data and person counts to a text file
def save_output_to_file(grouped_data, person_counts, car_types, output_file):
    try:
        with open(output_file, 'w') as f:
            for index, row in grouped_data.iterrows():
                depart = row['Depart']
                f.write(f"Depart: {depart}\nDestination: {row['Destination']}\nPersons: {row['Persons']}\nUnique Person Count: {person_counts[person_counts['Depart'] == depart]['Unique Person Count'].iloc[0]}\nCar Type Needed: {car_types[depart]}\n\n")

        print(f"Data saved to {output_file}")
    except Exception as e:
        print(f"Error saving the file: {e}")

class RouteOptimizer:
    def __init__(self, file_path, columns, coordinates, start_points, waypoints, end_points):
        self.file_path = file_path
        self.columns = columns
        self.coordinates = coordinates
        self.start_points = start_points
        self.waypoints = waypoints
        self.end_points = end_points
        self.df = None
        self.all_locations = None
        self.distances = {}
        self.shortest_paths = []

    def read_excel_columns(self):
        try:
            df = pd.read_excel(self.file_path)
            missing_columns = [col for col in self.columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Columns '{', '.join(missing_columns)}' not found in the Excel file.")
            self.df = df[self.columns]
        except Exception as e:
            print(f"Error reading the Excel file: {e}")
            self.df = None

    def process_data(self):
        if self.df is not None:
            self.df[['Depart', 'Destination']] = self.df['Text'].str.split(' to ', n=1, expand=True)
            self.df.dropna(inplace=True)
            self.all_locations = set(self.df['Depart'].tolist() + self.df['Destination'].tolist())
            self.compute_distances()

    def compute_distances(self):
        for loc1 in self.all_locations:
            for loc2 in self.all_locations:
                if loc1 != loc2 and loc1 in self.coordinates and loc2 in self.coordinates:
                    self.distances[(loc1, loc2)] = distance(self.coordinates[loc1], self.coordinates[loc2]).km

    def find_shortest_paths(self):
        valid_locations = [loc for loc in self.all_locations if loc in self.coordinates]
        remaining_locations = [loc for loc in valid_locations if loc not in self.start_points + self.waypoints + self.end_points]

        self.shortest_paths = []
        for start in self.start_points:
            for end in self.end_points:
                for perm in permutations(remaining_locations):
                    path = [start] + list(perm) + self.waypoints + [end]
                    total_distance = sum(self.distances.get((path[i], path[i+1]), float('inf')) for i in range(len(path) - 1))
                    self.shortest_paths.append((path, total_distance))

        self.shortest_paths.sort(key=lambda x: x[1])

    def generate_maps(self):
        map_objects = []
        for idx, (shortest_path, total_distance) in enumerate(self.shortest_paths):
            optimized_coordinates = [self.coordinates[loc] for loc in shortest_path]
            m = folium.Map(location=[optimized_coordinates[0][0], optimized_coordinates[0][1]], zoom_start=10)

            # Add markers and polyline for the optimized path
            for i in range(len(shortest_path) - 1):
                depart_condition = self.df['Depart'] == shortest_path[i]
                dest_condition = self.df['Destination'] == shortest_path[i + 1]
                if len(self.df.loc[depart_condition & dest_condition]) > 0:
                    persons_info = self.df.loc[depart_condition & dest_condition, 'Persons'].values[0]
                    count_persons = len(persons_info.split(', '))
                    popup_text = f'Coordinates: {optimized_coordinates[i + 1]}<br>Persons Count: {count_persons}<br>Persons: {persons_info}'
                    if shortest_path[i] in self.start_points:  # Check if current location is a start point
                        folium.Marker(optimized_coordinates[i + 1],
                                      popup=popup_text,
                                      icon=folium.Icon(color='red')).add_to(m)
                    else:
                        folium.Marker(optimized_coordinates[i + 1],
                                      popup=popup_text).add_to(m)

                    # Display information about persons on the map
                    display(f"For road segment from {shortest_path[i]} to {shortest_path[i + 1]}:\nPersons: {persons_info}\nDepart: {shortest_path[i]}\nDest: {shortest_path[i + 1]}\n Total Distance: {total_distance}\n\n")

                else:
                    print(f"No data found for journey from {shortest_path[i]} to {shortest_path[i + 1]}")

            folium.PolyLine(locations=optimized_coordinates, color='blue').add_to(m)

            # Add a car icon marker that moves along the polyline
            car_icon = folium.features.CustomIcon(
                icon_image='https://image.flaticon.com/icons/png/128/497/497555.png', icon_size=(30, 30))
            car_marker = folium.Marker(optimized_coordinates[0], icon=car_icon).add_to(m)

            # Function to animate the car marker along the path
            def animate_car():
                start_time = time.time()
                end_time = start_time + 10  # Adjust this value to control the duration (10 seconds in this case)
                for i in range(1, len(optimized_coordinates)):
                    # Update car marker position
                    car_marker.location = optimized_coordinates[i]

                    # Update polyline to show current path
                    folium.PolyLine(locations=optimized_coordinates[:i + 1], color='blue').add_to(m)

                    # Calculate remaining time in seconds
                    remaining_time = end_time - time.time()

                    # Break loop if remaining time is less than or equal to 0
                    if remaining_time <= 0:
                        break

                    # Delay for animation effect (adjust as needed)
                    time.sleep(remaining_time / (len(optimized_coordinates) - i))

                    # Clear previous car marker and add updated one
                    car_marker.add_to(m)

                    # Save map to HTML file (optional, uncomment if needed)
                    # m.save(f"map_{idx + 1}_frame_{i}.html")
                    
                    # Display the updated map
                    map_objects.append(m._repr_html_())

            # Animate the car marker along the path
            animate_car()

        return map_objects

    def print_debug_info(self):
        if self.shortest_paths:
            for path, total_distance in self.shortest_paths:
                print("Shortest Path:", path)
                print(f"Optimized path: {' -> '.join(path)}")
                print(f"Total distance: {total_distance} km")
                print("---------------------------------------------------------------------")

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file upload and process data
@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file.save(file.filename)  # Save the uploaded file
            # Process the uploaded file using your existing functions
            df_text = read_excel_column(file.filename, 'Text')
            df_persons = read_excel_column(file.filename, 'Persons')
            if df_text is not None and df_persons is not None:
                df_combined = pd.DataFrame({'Text': df_text, 'Persons': df_persons})
                grouped_data, person_counts, car_types = process_data(df_combined)
                if grouped_data is not None and person_counts is not None and car_types is not None:
                    output_file_path = 'output_depart_dest_persons.txt'
                    save_output_to_file(grouped_data, person_counts, car_types, output_file_path)
                    
                    # Create RouteOptimizer instance and process data
                    optimizer = RouteOptimizer(file.filename, ['Text', 'Persons'], coordinates, ['Hotel Mövenpick', 'SA'], ['Haupteingang DRX METS', 'KA'], ['airport TUN', 'DE'])
                    optimizer.read_excel_columns()
                    optimizer.process_data()
                    optimizer.find_shortest_paths()
                    map_objects = optimizer.generate_maps()
                    optimizer.print_debug_info()

                    return render_template('result.html', maps=map_objects)
                else:
                    return "Failed to process the data."
            else:
                return "Failed to read the column data."
    return "No file selected"

if __name__ == '__main__':
    app.run(debug=True)
