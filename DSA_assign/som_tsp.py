import numpy as np
import random
import math
import matplotlib.pyplot as plt

# Define the TSP graph (cities as coordinates)
cities = {1: (0, 0), 2: (1, 2), 3: (2, 4), 4: (3, 5), 5: (4, 1), 6: (5, 3), 7: (6, 6)}

# Number of cities
num_cities = len(cities)


# Function to calculate the Euclidean distance between two cities
def euclidean_distance(city1, city2):
    return np.linalg.norm(np.array(city1) - np.array(city2))


# Initialize the SOM grid: a 2D grid with neurons
grid_size = 5  # Grid size (5x5)
grid = np.random.rand(
    grid_size, grid_size, 2
)  # 2D grid with 2 coordinates for each neuron (x, y)

# SOM parameters
learning_rate_initial = 0.5
neighborhood_radius_initial = (
    grid_size / 2
)  # Start with the largest possible neighborhood
iterations = 1000  # Number of iterations


# Function to find the Best Matching Unit (BMU)
def find_bmu(city, grid):
    distances = np.linalg.norm(
        grid - city, axis=2
    )  # Compute distances between city and each neuron (correct axis)
    bmu_index = np.unravel_index(np.argmin(distances), distances.shape)
    return bmu_index


# Function to update the grid (SOM learning rule)
def update_grid(grid, city, bmu_index, learning_rate, neighborhood_radius):
    # Decay the neighborhood radius over time
    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate the distance between the current neuron and the BMU
            distance_to_bmu = np.linalg.norm(np.array([i, j]) - np.array(bmu_index))

            # If within the neighborhood radius, update the neuron position
            if distance_to_bmu <= neighborhood_radius:
                # Influence function (Gaussian-like)
                influence = math.exp(
                    -(distance_to_bmu**2) / (2 * (neighborhood_radius**2))
                )
                # Update the weight of the neuron
                grid[i, j] += learning_rate * influence * (city - grid[i, j])


# Training loop
learning_rate = learning_rate_initial
neighborhood_radius = neighborhood_radius_initial

for iteration in range(iterations):
    # Select a random city from the cities list
    city_id = random.choice(list(cities.keys()))
    city = np.array(cities[city_id])

    # Find the BMU for the current city
    bmu_index = find_bmu(city, grid)

    # Update the grid (neurons) based on BMU and the city
    update_grid(grid, city, bmu_index, learning_rate, neighborhood_radius)

    # Decay the learning rate and neighborhood radius
    learning_rate = learning_rate_initial * (1 - iteration / iterations)
    neighborhood_radius = neighborhood_radius_initial * (1 - iteration / iterations)


# Extracting the final route: cities ordered by their positions in the grid
def extract_route(grid, cities):
    route = []
    visited = set()  # Track which neurons we've visited

    # Iterate through the grid and find the closest city for each neuron
    for i in range(grid_size):
        for j in range(grid_size):
            # Find the nearest city to this grid position
            distances_to_cities = {
                city_id: euclidean_distance(cities[city_id], grid[i, j])
                for city_id in cities
            }
            nearest_city_id = min(distances_to_cities, key=distances_to_cities.get)

            # Append the city to the route if not visited
            if nearest_city_id not in visited:
                route.append(nearest_city_id)
                visited.add(nearest_city_id)

    return route


# Final route
route = extract_route(grid, cities)
print("Optimal Route:", route)

# Calculate total distance
total_distance = 0
for i in range(len(route) - 1):
    city1 = cities[route[i]]
    city2 = cities[route[i + 1]]
    total_distance += euclidean_distance(city1, city2)
total_distance += euclidean_distance(
    cities[route[-1]], cities[route[0]]
)  # Return to start city

print("Total Distance:", total_distance)

# Visualize the final route and cities
fig, ax = plt.subplots()
for city_id, (x, y) in cities.items():
    ax.scatter(x, y, label=f"City {city_id}")
    ax.text(x, y, f" {city_id}", fontsize=12, verticalalignment="bottom")

# Plot the route
for i in range(len(route) - 1):
    city1 = cities[route[i]]
    city2 = cities[route[i + 1]]
    ax.plot([city1[0], city2[0]], [city1[1], city2[1]], "k-")

# Return to the start city
ax.plot(
    [cities[route[-1]][0], cities[route[0]][0]],
    [cities[route[-1]][1], cities[route[0]][1]],
    "k-",
)

plt.title("SOM for TSP - Optimal Route")
plt.legend()
plt.grid(True)
plt.show()
