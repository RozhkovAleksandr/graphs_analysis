import numpy as np
from graphblas import Matrix, semiring, binary
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
        next_parents = front.mxm(graph_matrix, semiring.ss.any_secondi)

        not_visited = Matrix.full(bool, sources_number, n, True)
        not_visited = not_visited.ewise_add(visited, op=binary.minus)

        new_entries = next_parents.ewise_mult(not_visited)

        if new_entries.nvals == 0:
            break

        visited(mask=new_entries.S) << True
        parent(mask=new_entries.S) << new_entries

        front = Matrix(bool, sources_number, n)
        front(mask=new_entries.S) << True

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

    print(dense)


if __name__ == '__main__':
    main()
