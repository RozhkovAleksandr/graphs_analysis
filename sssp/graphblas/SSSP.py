import sys
from graphblas import Matrix, Vector, binary, semiring
import numpy as np


def load_graph(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    sources, targets, weights = [], [], []
    max_vertex = 0

    for line in lines:
        parts = line.split()
        s, t = int(parts[0]), int(parts[1])
        w = float(parts[2]) if len(parts) >= 3 else 1.0

        targets.append(t)
        sources.append(s)
        weights.append(w)

        if s > max_vertex:
            max_vertex = s
        if t > max_vertex:
            max_vertex = t

    n = max_vertex + 1
    A = Matrix(float, n, n)
    A.build(targets, sources, weights)

    return A

def sssp(A, source_vertex):
    n = A.nrows
    d = Vector(A.dtype, n)

    d[:] = float('inf')
    d[source_vertex] = 0

    for _ in range(n - 1):
        prev_d = d.dup()
        d(binary.min) << A.mxv(d, semiring.min_plus)

        if d.isequal(prev_d):
            break

    return d

def main():
    if len(sys.argv) < 3:
        print("Usage: sssp.py <edge_file> <start_vertex>")
        sys.exit(1)

    path = sys.argv[1]
    source = int(sys.argv[2])

    graph = load_graph(path)
    result = sssp(graph, source)
    
    print(result)

if __name__ == "__main__":
    main()