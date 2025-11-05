from copy import deepcopy

def gauss_jordan_solve(A, b, tol=1e-12):
    """
    Solve A x = b via Gauss-Jordan elimination (partial pivoting).
    Returns: x (list of floats), RREF of [A|b], and rank information.
    """
    # Build augmented matrix [A | b]
    n = len(A)
    m = len(A[0])
    assert len(b) == n, "Dimension mismatch: len(b) must equal number of rows in A"
    aug = [list(A[i]) + [b[i]] for i in range(n)]

    row = 0
    pivots = []
    for col in range(m):
        # Find pivot (max abs value in current column from 'row' downwards)
        pivot = max(range(row, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < tol:
            continue  # no pivot in this column
        # Swap to top
        aug[row], aug[pivot] = aug[pivot], aug[row]

        # Scale pivot row to make pivot = 1
        piv = aug[row][col]
        aug[row] = [v / piv for v in aug[row]]

        # Eliminate other rows
        for r in range(n):
            if r == row:
                continue
            factor = aug[r][col]
            if abs(factor) > tol:
                aug[r] = [aug[r][c] - factor * aug[row][c] for c in range(m+1)]

        pivots.append((row, col))
        row += 1
        if row == n:
            break

    # Extract solution if consistent
    # Check for inconsistency: a row like [0,0,...,0 | nonzero]
    for r in range(n):
        if all(abs(aug[r][c]) < tol for c in range(m)) and abs(aug[r][m]) > tol:
            raise ValueError("System is inconsistent: no solution.")

    # If system underdetermined, this routine gives one solution (free vars = 0)
    # Read solution by columns that became pivots
    x = [0.0] * m
    for r, c in pivots:
        x[c] = aug[r][m]

    return x, aug, len(pivots)


def rref(A, tol=1e-12):
    """
    Reduced Row Echelon Form (RREF) of matrix A (without b).
    """
    M = deepcopy(A)
    n = len(M)
    m = len(M[0]) if n > 0 else 0

    row = 0
    for col in range(m):
        # pivot search
        pivot = max(range(row, n), key=lambda r: abs(M[r][col]))
        if n == 0 or abs(M[pivot][col]) < tol:
            continue
        # swap
        M[row], M[pivot] = M[pivot], M[row]
        # scale
        piv = M[row][col]
        M[row] = [v/piv for v in M[row]]
        # eliminate
        for r in range(n):
            if r == row:
                continue
            factor = M[r][col]
            if abs(factor) > tol:
                M[r] = [M[r][c] - factor*M[row][c] for c in range(m)]
        row += 1
        if row == n:
            break
    return M


def inverse_via_gauss_jordan(A, tol=1e-12):
    """
    Compute inverse of a square matrix A using Gauss-Jordan on [A | I].
    Raises ValueError if not invertible (singular).
    """
    n = len(A)
    assert all(len(row) == n for row in A), "A must be square"
    # Build augmented [A | I]
    aug = [list(A[i]) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    row = 0
    for col in range(n):
        # pivot search
        pivot = max(range(row, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < tol:
            raise ValueError("Matrix is singular; inverse does not exist.")
        # swap
        aug[row], aug[pivot] = aug[pivot], aug[row]
        # scale pivot row
        piv = aug[row][col]
        aug[row] = [v/piv for v in aug[row]]
        # eliminate
        for r in range(n):
            if r == row:
                continue
            factor = aug[r][col]
            if abs(factor) > tol:
                aug[r] = [aug[r][c] - factor*aug[row][c] for c in range(2*n)]
        row += 1

    # Right half is A^{-1}
    inv = [row[n:] for row in aug]
    return inv
