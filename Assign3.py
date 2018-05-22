import numpy as np
import pandas as pd
import string
import sys
import itertools

# Conduct Adjacency Matrix
undi_node = np.array([[0,1,0,1,1],[1,0,1,1,0],[0,1,0,1,0],[1,1,1,0,1],[1,0,0,1,0]]).reshape(5,5) # Symmetric Metrix
di_node = np.array([[0,1,0,0,0],[0,0,1,1,0],[0,1,0,0,0],[1,0,1,0,1],[1,0,0,0,0]]).reshape(5,5)

# Question 01
# Degree Centrality

# Case::Undirected Graph

def undi_degree_centrality(adjacencyMatrix):
    '''
    Calculating a degree centrality for undirected graph
    :param adjacencyMatrix: Maximum 26 by 26 because of key(alphabet is 26 characters)
    :return: result of Dictionary for each node
    '''

    df_undi = pd.DataFrame(adjacencyMatrix)
    degree = np.sum(adjacencyMatrix, axis=1)
    df_deg = pd.DataFrame(degree, columns=['degree'])

    df = pd.concat([df_undi, df_deg], axis=1)
    '''
    df.rename(columns={0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'},
              index={0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}, inplace=True)

    result = {'A':df['degree'][0] / (len(degree) - 1),
              'B':df['degree'][1] / (len(degree) - 1),
              'C':df['degree'][2] / (len(degree) - 1),
              'D':df['degree'][3] / (len(degree) - 1),
              'E':df['degree'][4] / (len(degree) - 1)}
    '''
    deg_cen = df['degree'] / (len(degree) - 1)
    node = string.ascii_uppercase[:len(degree)]

    result = {k:v for k,v in zip(node, deg_cen.tolist())}

    return result

# Case::Directed Graph

def di_in_degree_centrality(adjacencyMatrix):
    '''
    Calculating a in_degree centrality for directed graph
    :param adjacencyMatrix: Maximum 26 by 26 because of key(alphabet is 26 characters)
    :return: result of Dictionary for each node
    '''

    df_di = pd.DataFrame(adjacencyMatrix)
    indegree = np.sum(adjacencyMatrix, axis=0)
    df_deg = pd.DataFrame(indegree, columns=['indegree'])

    df = pd.concat([df_di, df_deg], axis=1)

    deg_cen = df['indegree'] / (len(indegree) - 1)
    node = string.ascii_uppercase[:len(indegree)]

    result = {k:v for k,v in zip(node, deg_cen.tolist())}

    return result

def di_out_degree_centrality(adjacencyMatrix):
    '''
    Calculating a out_degree centrality for directed graph
    :param adjacencyMatrix: Maximum 26 by 26 because of key(alphabet is 26 characters)
    :return: result of Dictionary for each node
    '''

    df_di = pd.DataFrame(adjacencyMatrix)
    outdegree = np.sum(adjacencyMatrix, axis=1)
    df_deg = pd.DataFrame(outdegree, columns=['outdegree'])

    df = pd.concat([df_di, df_deg], axis=1)

    deg_cen = df['outdegree'] / (len(outdegree) - 1)
    node = string.ascii_uppercase[:len(outdegree)]

    result = {k:v for k,v in zip(node, deg_cen.tolist())}

    return result

#print('degree centrality: ',undi_degree_centrality(undi_node))
#print('in-degree centrality: ',di_in_degree_centrality(di_node))
#print('out-degree centrality: ',di_out_degree_centrality(di_node))


# Question 02
# Closeness centrality

# dijkstra's algorithm for shortest path length
class Graph():
    '''
    This class for calculating a dijkstra shortest path length using adjacency matrix
    '''

    def __init__(self, vertex):
        self.vertex = vertex
        self.graph = [[0 for column in range(vertex)] for row in range(vertex)]


    def minDistance(self, distance, shortestPathbool):

        # Initilaize
        inf = sys.maxsize

        # shortest path tree
        for v in range(self.vertex):
            if distance[v] < inf and shortestPathbool[v] == False:
                inf = distance[v]
                min_index = v

        return min_index


    def dijkstra(self, search):

        distance = [sys.maxsize] * self.vertex
        distance[search] = 0
        shortestPathbool = [False] * self.vertex

        for count in range(self.vertex):

            u = self.minDistance(distance, shortestPathbool)

            shortestPathbool[u] = True

            # Update distance
            for v in range(self.vertex):
                if self.graph[u][v] > 0 and shortestPathbool[v] == False and distance[v] > distance[u] + self.graph[u][v]:
                    distance[v] = distance[u] + self.graph[u][v]

        return distance


test_undi = np.array([[0,1,None,1,1],[1,0,1,1,None],[None,1,0,1,None],[1,1,1,0,1],[1,None,None,1,0]]).reshape(5,5)
test_di = np.array([[0,1,None,None,None],[None,0,1,1,None],[None,1,0,None,None],[1,None,1,0,1],[1,None,None,None,0]]).reshape(5,5)

test = [[0,3,2,4,None,None],
        [3,0,None,2,None,5],
        [2,None,0,None,1,None],
        [4,2,None,0,1,3],
        [None,None,1,1,0,2],
        [None,5,None,3,2,0]]

def dijkstra_shortest_path(graph, search):
    """
    Performs the shortest path algorithm and returns a list of vertexes in order with their distances from the start.
    """
    distances_from_start = [None] * len(graph)

    visited_vertexes = []

    current_vertex = 0

    distances_from_start[current_vertex] = [0, 0]  # [distance from start, via vertex]

    for row in range(len(graph)):

        current_vertex = row

        #print("Current vertex: ", current_vertex)

        # Iterate through each column in the current row in the adjacency matrix
        for col in range(len(graph[current_vertex])):

            if graph[current_vertex][col] is not None and distances_from_start[col] is None:
                distances_from_start[col] = [distances_from_start[current_vertex][0] + graph[current_vertex][col], current_vertex]

            elif graph[current_vertex][col] is not None and (graph[current_vertex][col] + distances_from_start[current_vertex][0]) < distances_from_start[col][0]:
                distances_from_start[col] = [(graph[current_vertex][col] + distances_from_start[current_vertex][0]), current_vertex]

        print("Distances from start: ", distances_from_start)  # show updated distances_from_start array

        # Add current_vertex to visited list so that its distance from the start is calculated again in future
        if current_vertex not in visited_vertexes:
            visited_vertexes.append(current_vertex)

        # print("Visited vertexes: ", visited_vertexes)

    # Print the shortest path in a friendly format
    print("Shortest path:")
    current_vertex = search #len(graph) - 1
    path_string = ""
    orderlist = []
    while current_vertex > 0:

        # Add the distance for the current vertex from the start in brackets after the letter of the vertex.
        path_string = "{0}({1}) ".format(chr(current_vertex + 65), distances_from_start[current_vertex][0]) + path_string

        temp = [chr(current_vertex + 65), distances_from_start[current_vertex][0]]

        orderlist.append(temp)

        # Update the current vertex to be the one that the current one goes via on its way back to the start
        current_vertex = distances_from_start[current_vertex][1]  # distances_from_start[vertex number, via vertex]


    # Add the start vertex to the output string as the while loop will stop before we add its details to the string
    path_string = "{0}({1}) ".format(chr(current_vertex + 65), distances_from_start[current_vertex][0]) + path_string

    temp = [chr(current_vertex + 65), distances_from_start[current_vertex][0]]
    orderlist.append(temp)

    print(path_string)

    return orderlist[::-1]

print(dijkstra_shortest_path(test_undi, 2))

def closeness_centrality(size ,adjacencyMatrix, start):
    '''
    Calculating the closeness centrality using adjacency matrix
    :param size: the number of nodes
    :param adjacencyMatrix: expression from graph to matrix
    :param start: starting node
    :return: float, closeness centrality
    '''

    g = Graph(size)
    g.graph = adjacencyMatrix
    g.dijkstra(start)
    print(g.dijkstra(start))

    return (size-1)/np.sum(g.dijkstra(start))

#print(closeness_centrality(5,di_node,0))


# Question 03
# Betweeness centrality

def betweeness_centrality(graph, node, include = False):

    index = [i for i in range(len(graph))]
    index.remove(node)

    print(list(itertools.combinations(index, 2)))


print(betweeness_centrality(undi_node,0))


