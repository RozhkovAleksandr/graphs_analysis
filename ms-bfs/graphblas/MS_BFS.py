import numpy as np
from graphblas import Matrix, Vector, semiring, binary
import argparse


def ms_bfs_parent(graph_matrix, sources):
    n = graph_matrix.nrows
    sources_number = len(sources)

    front = Matrix(bool, sources_number, n)
    visited = Matrix(bool, sources_number, n)
    parent = Matrix(int, sources_number, n)

    for i, source in enumerate(sources):
        front[i, source] = True
        visited[i, source] = True
        parent[i, source] = source

    while front.nvals > 0:
        next_parents = front.mxm(graph_matrix, semiring.any_secondi)

        not_visited = Matrix(bool, sources_number, n)
        not_visited[:, :] = 1
        not_visited = not_visited.ewise_add(visited, op=binary.minus)

        next_parents = next_parents.select(not_visited)

        if next_parents.nvals == 0:
            break

        visited(mask=next_parents.S) << True
        parent(mask=next_parents.S) << next_parents

        front = Matrix(bool, sources_number, n)
        front(mask=next_parents.S) << True

    return parent


def main():
    parser = argparse.ArgumentParser(
        description="Multi-source BFS with parent tracking using GraphBLAS."
    )
    parser.add_argument('graph_file', help='Path to tab-delimited edge list')
    parser.add_argument('sources', nargs='+', type=int,
                        help='List of starting vertices (0-based)')
    args = parser.parse_args()

    edges = np.loadtxt(args.graph_file, dtype=int, delimiter=None)
    u, v = edges[:, 0].tolist(), edges[:, 1].tolist()

    row_indices = u + v
    col_indices = v + u

    adj = Matrix.from_coo(row_indices, col_indices, True)

    parent = ms_bfs_parent(adj, args.sources)
    dense = parent.to_dense(fill_value=-1)


if __name__ == '__main__':
    main()