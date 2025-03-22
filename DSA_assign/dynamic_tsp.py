from functools import lru_cache


def make_full_graph(graph):
    nodes = list(graph.keys())
    full_graph = {node: {} for node in nodes}

    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            full_graph[i][j] = graph[i].get(j, float("inf"))

    return full_graph


def tsp_dynamic_programming(graph, start):
    graph = make_full_graph(graph)  # Ensure full connectivity
    n = len(graph)
    all_visited = (1 << n) - 1  # All cities visited

    @lru_cache(None)
    def visit(city, visited):
        if visited == all_visited:  # All cities visited
            return graph[city].get(start, float("inf")), [start]  # Return to start

        min_distance = float("inf")
        best_route = []

        for next_city in graph:
            bit = 1 << (next_city - 1)  # Adjust bitmasking for 1-based index
            if next_city != start and not (visited & bit):  # If not visited
                if graph[city].get(next_city, float("inf")) == float("inf"):
                    continue  # Skip unreachable cities

                new_visited = visited | bit
                distance, route = visit(next_city, new_visited)
                total_distance = graph[city][next_city] + distance

                if total_distance < min_distance:
                    min_distance = total_distance
                    best_route = [next_city] + route

        return min_distance, best_route

    min_distance, best_route = visit(start, 1 << (start - 1))
    return (
        min_distance,
        [start] + best_route,
        min_distance,
    )  # Include the starting city and total cost


# Define the adjacency list graph representation
graph = {
    1: {2: 12, 3: 10, 7: 12},
    2: {1: 12, 3: 8, 4: 12},
    3: {1: 10, 2: 8, 4: 11, 5: 3, 7: 9},
    4: {2: 12, 3: 11, 5: 11, 6: 10},
    5: {3: 3, 4: 11, 6: 6, 7: 7},
    6: {4: 10, 5: 6, 7: 9},
    7: {1: 12, 3: 9, 5: 7, 6: 9},
}

# Run the TSP dynamic programming algorithm
start_city = 1
min_distance, best_route, total_cost = tsp_dynamic_programming(graph, start_city)

# Print the minimum distance and the best route
print("Total Distance:", min_distance)
print("Optimal Route:", " -> ".join(map(str, best_route)))
print("Total Route Cost:", total_cost)
