# Python program for Dijkstra's single
# source shortest path algorithm. The program is
# for adjacency matrix representation of the graph

# Library for INT_MAX
import sys


class Graph():

    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)] for row in range(vertices)]

    def printSolution(self, distance):
        print("Vertex | Distance from Source")
        for node in range(self.V):
            print(node, "  |  ", distance[node])

    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    def minDistance(self, distance, sptSet):

        # Initilaize minimum distance for next node
        inf = sys.maxsize

        # Search not nearest vertex not in the
        # shortest path tree
        for v in range(self.V):
            if distance[v] < inf and sptSet[v] == False:
                inf = distance[v]
                min_index = v

        return min_index

    # Funtion that implements Dijkstra's single source
    # shortest path algorithm for a graph represented
    # using adjacency matrix representation
    def dijkstra(self, search):

        distance = [sys.maxsize] * self.V
        distance[search] = 0
        sptSet = [False] * self.V

        for count in range(self.V):

            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to search in first iteration
            u = self.minDistance(distance, sptSet)

            # Put the minimum distance vertex in the
            # shotest path tree
            sptSet[u] = True

            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shotest path tree
            for v in range(self.V):
                if self.graph[u][v] > 0 and sptSet[v] == False and distance[v] > distance[u] + self.graph[u][v]:
                    distance[v] = distance[u] + self.graph[u][v]

        self.printSolution(distance)


if __name__ == '__main__':
    g = Graph(5)
    print(g.graph)
    g.graph = [[0,1,0,1,1],
               [1,0,1,1,0],
               [0,1,0,1,0],
               [1,1,1,0,1],
               [1,0,0,1,0]]
    g.dijkstra(1)