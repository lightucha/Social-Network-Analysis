graph = [[0,3,2,4,None,None],
        [3,0,None,2,None,5],
        [2,None,0,None,1,None],
        [4,2,None,0,1,3],
        [None,None,1,1,0,2],
        [None,5,None,3,2,0]]

distances_from_start = [None] * len(graph)

visited_vertexes = []


def print_adj_matrix():
    """ Prints the Adjacency Matrix representation of the global graph matrix in a nicely formatted way."""

    global graph

    print("\nAdjacency Matrix of Graph:\n")

    # Generate column headings
    headingString = "  "
    for colNum in range(len(graph)):
        headingString += "|{:^6}".format(chr(65 + colNum))
    print(headingString)
    print("-" * len(headingString))

    # Print each row
    for rowNum in range(len(graph)):
        rowString = "{0} ".format(chr(65 + rowNum))
        for colValue in graph[rowNum]:
            rowString += '|{:^6}'.format(str(colValue) if colValue is not None else "-")
        print(rowString)

    print("-" * len(headingString))


def shortest_path():
    """
    Performs the shortest path algorithm and returns a list of vertexes in order with their distances from the start.
    """
    global graph, distances_from_start, visited_vertexes

    current_vertex = 0

    distances_from_start[current_vertex] = [0, 0]  # [distance from start, via vertex]

    for rowNum in range(len(graph)):

        current_vertex = rowNum

        print("Current vertex: ", current_vertex)

        # Iterate through each column in the current row in the adjacency matrix
        for colNum in range(len(graph[current_vertex])):

            if graph[current_vertex][colNum] is not None and distances_from_start[colNum] is None:
                distances_from_start[colNum] = [distances_from_start[current_vertex][0] + graph[current_vertex][colNum], current_vertex]

            elif graph[current_vertex][colNum] is not None and (graph[current_vertex][colNum] + distances_from_start[current_vertex][0]) < distances_from_start[colNum][0]:
                distances_from_start[colNum] = [(graph[current_vertex][colNum] + distances_from_start[current_vertex][0]), current_vertex]

        print("Distances from start: ", distances_from_start)  # show updated distances_from_start array

        # Add current_vertex to visited list so that its distance from the start is calculated again in future
        if current_vertex not in visited_vertexes:
            visited_vertexes.append(current_vertex)

        # print("Visited vertexes: ", visited_vertexes)

    # Print the shortest path in a friendly format
    print("Shortest path:")
    current_vertex = len(graph) - 1
    path_string = ""
    temp = []
    while current_vertex > 0:

        # Add the distance for the current vertex from the start in brackets after the letter of the vertex.
        path_string = "{0}({1}) ".format(chr(current_vertex + 65), distances_from_start[current_vertex][0]) + path_string
        print(path_string)

        a = [chr(current_vertex + 65), distances_from_start[current_vertex][0]]

        temp.append(a)

        # Update the current vertex to be the one that the current one goes via on its way back to the start
        current_vertex = distances_from_start[current_vertex][1]  # distances_from_start[vertex number, via vertex]
        print(current_vertex)


    # Add the start vertex to the output string as the while loop will stop before we add its details to the string
    path_string = "{0}({1}) ".format(chr(current_vertex + 65), distances_from_start[current_vertex][0]) + path_string

    a = [chr(current_vertex + 65), distances_from_start[current_vertex][0]]
    temp.append(a)

    print(path_string)
    print('path:  ', temp[::-1])

#print_adj_matrix()
shortest_path()