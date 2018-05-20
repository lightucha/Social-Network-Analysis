import numpy as np
import pandas as pd
import string
import sys

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
    '''
    df.rename(columns={0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'},
              index={0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}, inplace=True)
    
    result = {'A':df['indegree'][0] / (len(indegree) - 1),
              'B':df['indegree'][1] / (len(indegree) - 1),
              'C':df['indegree'][2] / (len(indegree) - 1),
              'D':df['indegree'][3] / (len(indegree) - 1),
              'E':df['indegree'][4] / (len(indegree) - 1)}
    '''
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
    '''
    df.rename(columns={0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'},
              index={0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}, inplace=True)
    
    result = {'A':df['outdegree'][0] / (len(outdegree) - 1),
              'B':df['outdegree'][1] / (len(outdegree) - 1),
              'C':df['outdegree'][2] / (len(outdegree) - 1),
              'D':df['outdegree'][3] / (len(outdegree) - 1),
              'E':df['outdegree'][4] / (len(outdegree) - 1)}
    '''
    deg_cen = df['outdegree'] / (len(outdegree) - 1)
    node = string.ascii_uppercase[:len(outdegree)]

    result = {k:v for k,v in zip(node, deg_cen.tolist())}

    return result

print('degree centrality: ',undi_degree_centrality(undi_node))
print('in-degree centrality: ',di_in_degree_centrality(di_node))
print('out-degree centrality: ',di_out_degree_centrality(di_node))


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

def closeness_centrality(size ,adjacencyMatrix, start):

    g = Graph(size)
    g.graph = adjacencyMatrix
    g.dijkstra(start)
    print(g.dijkstra(start))

    return (size-1)/np.sum(g.dijkstra(start))

print(closeness_centrality(5,di_node,0))


# Question 03
# Betweeness centrality
