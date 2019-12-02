import numpy as np


def readFile(filename = 'web-Google.txt'):
    vtx_map = dict()
    edge_list = []
    f = open(filename)
    lines = f.readlines()[4:]
    print('read lines: ', len(lines))
    for line in lines:
        tmp = line.strip().split('\t')
        assert(len(tmp) == 2), 'current line: %s'%tmp
        # read_content = {}
        # read_content['start'] = tmp[0]
        # read_content['end'] = tmp[1]
        # content.append(read_content)
        start = add_vertex(tmp[0], vtx_map)
        end = add_vertex(tmp[1], vtx_map)
        edge = Edge(start, end)
        edge_list.append(edge)
        # print(edge)
    return vtx_map, edge_list


def add_vertex(vertex, vtx_map):
    id = 0
    if vertex in vtx_map:
        return vtx_map[vertex]
    else:
        id = len(vtx_map)
        vtx_map[vertex] = id
    return id


class Vertex:
    def __init__(self):
        self.in_node = 0
        self.out_node = 0
        self.score = 0.0


class Edge:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __str__(self):
        return "%s -> %s"%(self.start, self.end)


def initiation(vertices, edges):
    num_vertex = len(vertices)
    num_edges = len(edges)
    M = np.zeros((num_vertex, num_vertex))
    for edge in edges:
        M[int(edge.end)][int(edge.start)] = 1
    b = np.transpose(M)
    result_M = np.zeros((M.shape),dtype=float)
    # for j in range(np.shape(b)[1]):
    #     print(b[j].sum())
    for i in range(np.shape(M)[0]):
        for j in range(np.shape(M)[1]):
            if b[j].sum() != 0:
                result_M[i][j] = M[i][j] / (b[j].sum())
    init_rank = np.zeros((np.shape(M)[0], 1), dtype=float)
    for i in range(np.shape(init_rank)[0]):
        init_rank[i] = 1.0 / num_vertex
    return result_M, init_rank


def updatePageRank(beta, M, rank):
    M = np.mat(M)
    rank = np.mat(rank)
    num_verties = np.shape(M)[0]
    iter = 1
    while np.sum(abs(rank - (beta*np.matmul(M,rank) + (1-beta)*1/num_verties))) > 0.0001:
        print(iter)
        rank = beta * np.matmul(M, rank) + (1 - beta) * 1/num_verties
        iter += 1
    return rank

if __name__ == '__main__':
    vtx_list, edge_list = readFile()
    print(vtx_list)
    print(edge_list)
    M, init_rank = initiation(vtx_list, edge_list)
    # print(M)
    # print(init_rank)
    rank = updatePageRank(0.90, M, init_rank)
    print(rank)