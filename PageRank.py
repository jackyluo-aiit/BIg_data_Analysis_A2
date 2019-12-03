import numpy as np
import sys
from scipy import sparse


def readFile(vtx_map, filename='web-Google.txt'):
    edge_list = []
    f = open(filename)
    lines = f.readlines()[4:]
    print('read lines: ', len(lines))
    for line in lines:
        tmp = line.strip().split('\t')
        assert (len(tmp) == 2), 'current line: %s' % tmp
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
        return "%s -> %s" % (self.start, self.end)


def initiation(vertices, edges):
    num_vertex = len(vertices)
    num_edges = len(edges)

    #sparse implementation
    vtx_list = [Vertex() for _ in range(num_vertex)]
    row = list()
    col = list()
    data = list()
    # M = np.zeros((num_vertex, num_vertex))
    for edge in edges:
        # M[int(edge.end)][int(edge.start)] = 1
        row.append(int(edge.end))
        col.append(int(edge.start))
        vtx_list[edge.start].out_node += 1
        vtx_list[edge.end].in_node += 1
    for edge in edges:
        data.append(1/vtx_list[edge.start].out_node)
    result_M = sparse.coo_matrix((data, (row, col)), shape=(num_vertex, num_vertex))
    print('Trans_matrix completed!!!')

    # dense implementation
    # b = np.transpose(M)
    # result_M = np.zeros((M.shape), dtype=float)
    # # for j in range(np.shape(b)[1]):
    # #     print(b[j].sum())
    # for i in range(np.shape(M)[0]):
    #     for j in range(np.shape(M)[1]):
    #         if b[j].sum() != 0:
    #             result_M[i][j] = M[i][j] / (b[j].sum())

    init_rank = np.zeros((np.shape(result_M)[0], 1), dtype=float)
    for i in range(np.shape(init_rank)[0]):
        init_rank[i] = 1.0 / num_vertex
    print('Intial rank completed!!!')
    return result_M, init_rank


def updatePageRank(beta, M, rank):
    # dense implementation
    # # M = np.mat(M)
    rank = np.mat(rank)

    # sparse implementation
    M = M.todense()
    # rank = sparse.csr_matrix(rank)
    num_verties = np.shape(M)[0]
    iter = 1
    while np.sum(abs(rank - (beta * np.matmul(M, rank) + (1 - beta) * 1 / num_verties))) > 0.0001:
        print('iteration:%d' % iter)
        rank = beta * np.matmul(M, rank) + (1 - beta) * 1 / num_verties
        iter += 1
    return rank


def reverse_map(rank, vtx_map):
    result_node = []
    for index in rank:
        if index in vtx_map.values():
            result_node.append(list(vtx_map.keys())[list(vtx_map.values()).index(index)])
    return result_node


def evluation(file1, file2):
    list1 = list(np.loadtxt(file1))
    list2 = list(np.loadtxt(file2))
    count = 0
    assert len(list1) == len(list2), 'Two lists are in different length!!!'
    for each in range(len(list1)):
        if list2[each] != list1[each]:
            count += 1
    return count/len(list1)


if __name__ == '__main__':
    # print(evluation('result_test2_new.txt', 'result_test2_old.txt'))
    # sys.exit(0)
    vtx_map = dict()
    file = 'web-Google.txt'
    try:
        vtx_list, edge_list = readFile(vtx_map, file)
    except FileNotFoundError:
        print("File %s not found!!!" % file)
        sys.exit(1)
    print('Read file completed!!!')
    M, init_rank = initiation(vtx_list, edge_list)
    print('Initiation completed!!!')
    # print(M.toarray())
    # print(init_rank)
    rank = updatePageRank(0.90, M, init_rank)
    print('Updated rank!!!')
    # print(rank)
    rank = np.squeeze(np.array(rank), axis=1)
    rank = np.argsort(-rank)
    rank = reverse_map(rank, vtx_map)
    print(rank[:100])
    np.savetxt('result.txt', np.array(rank), fmt="%s")
