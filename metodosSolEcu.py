import numpy as np
from abc import ABC, abstractmethod


class sistema_ecuaciones(ABC):
    @abstractmethod
    def solve(self, A, b):
        pass



class GaussianElimination(sistema_ecuaciones):
    """ Eliminación Gaussiana con Pivoteo Parcial (Estándar) """
    def solve(self, A, b):
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        n = len(b)
        if A.shape != (n, n): raise ValueError("Matriz no cuadrada")

        for k in range(n - 1):
            max_index = np.argmax(np.abs(A[k:n, k])) + k
            if A[max_index, k] == 0: raise ValueError("Matriz singular")
            if max_index != k:
                A[[k, max_index]] = A[[max_index, k]]
                b[[k, max_index]] = b[[max_index, k]]
            for i in range(k + 1, n):
                factor = A[i, k] / A[k, k]
                A[i, k:] -= factor * A[k, k:]
                b[i] -= factor * b[k]
        
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            sum_j = np.dot(A[i, i+1:], x[i+1:])
            x[i] = (b[i] - sum_j) / A[i, i]
        return x

class GaussJordanElimination(sistema_ecuaciones):
    """ Eliminación Gauss-Jordan """
    def solve(self, A, b):
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        n = len(b)
        Aug = np.column_stack((A, b))

        for k in range(n):
            max_index = np.argmax(np.abs(Aug[k:n, k])) + k
            if Aug[max_index, k] == 0: raise ValueError("Matriz singular")
            Aug[[k, max_index]] = Aug[[max_index, k]]
            Aug[k] = Aug[k] / Aug[k, k]
            for i in range(n):
                if i != k:
                    Aug[i] -= Aug[i, k] * Aug[k]
        return Aug[:, -1]

class GaussianTotalPivoting(sistema_ecuaciones):
    """ Eliminación Gaussiana con Pivoteo Total """
    def solve(self, A, b):
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        n = len(b)
        col_swaps = []

        for k in range(n - 1):
            submatrix = np.abs(A[k:, k:])
            max_val_idx = np.unravel_index(np.argmax(submatrix, axis=None), submatrix.shape)
            max_row = max_val_idx[0] + k
            max_col = max_val_idx[1] + k
            
            if A[max_row, max_col] == 0: raise ValueError("Matriz singular")

            if max_row != k:
                A[[k, max_row]] = A[[max_row, k]]
                b[[k, max_row]] = b[[max_row, k]]
            
            if max_col != k:
                A[:, [k, max_col]] = A[:, [max_col, k]]
                col_swaps.append((k, max_col))

            for i in range(k + 1, n):
                factor = A[i, k] / A[k, k]
                A[i, k:] -= factor * A[k, k:]
                b[i] -= factor * b[k]

        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            sum_j = np.dot(A[i, i+1:], x[i+1:])
            x[i] = (b[i] - sum_j) / A[i, i]

        for k, max_col in reversed(col_swaps):
            x[[k, max_col]] = x[[max_col, k]]
        return x


class LUDoolittle(sistema_ecuaciones):
    """ 1. Factorización LU (Doolittle): 1s en la diagonal de L """
    def solve(self, A, b):
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        n = len(b)
        L = np.eye(n)
        U = np.zeros((n, n))

        # Doolittle Algorithm
        for i in range(n):
            # Upper Triangular
            for k in range(i, n):
                sum_u = sum(L[i][j] * U[j][k] for j in range(i))
                U[i][k] = A[i][k] - sum_u
            # Lower Triangular
            for k in range(i + 1, n):
                sum_l = sum(L[k][j] * U[j][i] for j in range(i))
                if U[i][i] == 0: raise ValueError("Pivote 0 encontrado (requiere pivoteo)")
                L[k][i] = (A[k][i] - sum_l) / U[i][i]

        # Resolver Ly = b
        y = np.zeros(n)
        for i in range(n):
            y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))
            
        # Resolver Ux = y
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
        return x

class LUCrout(sistema_ecuaciones):
    """ 2. Factorización LU (Crout): 1s en la diagonal de U """
    def solve(self, A, b):
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        n = len(b)
        L = np.zeros((n, n))
        U = np.eye(n)

        for j in range(n):
            # Columna de L
            for i in range(j, n):
                sum_l = sum(L[i][k] * U[k][j] for k in range(j))
                L[i][j] = A[i][j] - sum_l
            
            # Fila de U
            for i in range(j + 1, n):
                sum_u = sum(L[j][k] * U[k][i] for k in range(j))
                if L[j][j] == 0: raise ValueError("Pivote 0 encontrado")
                U[j][i] = (A[j][i] - sum_u) / L[j][j]

        # Resolver Ly = b
        y = np.zeros(n)
        for i in range(n):
            y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]

        # Resolver Ux = y
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))
        return x

class CholeskyFactorization(sistema_ecuaciones):
    """ 3. Factorización Cholesky (LL^T): Para matrices Simétricas Definidas Positivas """
    def solve(self, A, b):
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        n = len(b)
        
        # Comprobar simetría
        if not np.allclose(A, A.T):
            raise ValueError("La matriz no es simétrica. Cholesky requiere simetría.")
            
        L = np.zeros((n, n))

        try:
            for i in range(n):
                for j in range(i + 1):
                    sum_k = sum(L[i][k] * L[j][k] for k in range(j))
                    if i == j:
                        val = A[i][i] - sum_k
                        if val <= 0: raise ValueError("La matriz no es definida positiva")
                        L[i][j] = np.sqrt(val)
                    else:
                        L[i][j] = (A[i][j] - sum_k) / L[j][j]
        except Exception:
            raise ValueError("Error calculando Cholesky (¿Matriz def. positiva?)")

        # Resolver Ly = b
        y = np.zeros(n)
        for i in range(n):
            y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]
            
        # Resolver L.T x = y
        L_T = L.T
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - sum(L_T[i][j] * x[j] for j in range(i + 1, n))) / L_T[i][i]
        return x

class QRFactorization(sistema_ecuaciones):
    """ 4. Factorización QR: A = QR, Rx = Q.T b """
    def solve(self, A, b):
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        n = len(b)
        
        # Gram-Schmidt para obtener Q y R
        Q = np.zeros((n, n))
        R = np.zeros((n, n))
        
        for j in range(n):
            v = A[:, j]
            for i in range(j):
                R[i, j] = np.dot(Q[:, i], A[:, j])
                v = v - R[i, j] * Q[:, i]
            R[j, j] = np.linalg.norm(v)
            if R[j, j] == 0: raise ValueError("Dependencia lineal detectada (Singular)")
            Q[:, j] = v / R[j, j]
            
        # Resolver Rx = y (donde y = Q.T * b)
        y = np.dot(Q.T, b)
        
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - sum(R[i][j] * x[j] for j in range(i + 1, n))) / R[i][i]
        return x


class Jacobi(sistema_ecuaciones):
    def __init__(self, max_iter=100, tolerance=1e-6):
        self.max_iter = max_iter
        self.tolerance = tolerance
    def solve(self, A, b):
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        n = len(b)
        x = np.zeros(n)
        x_new = np.zeros(n)
        for k in range(self.max_iter):
            for i in range(n):
                s1 = np.dot(A[i, :i], x[:i])
                s2 = np.dot(A[i, i+1:], x[i+1:])
                x_new[i] = (b[i] - s1 - s2) / A[i, i]
            if np.linalg.norm(x_new - x) < self.tolerance: return x_new
            x = x_new.copy()
        raise RuntimeError("No convergió")

class GaussSeidel(sistema_ecuaciones):
    def __init__(self, max_iter=100, tolerance=1e-6):
        self.max_iter = max_iter
        self.tolerance = tolerance
    def solve(self, A, b):
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        n = len(b)
        x = np.zeros(n)
        for k in range(self.max_iter):
            x_old = x.copy()
            for i in range(n):
                s1 = np.dot(A[i, :i], x[:i])
                s2 = np.dot(A[i, i+1:], x_old[i+1:])
                x[i] = (b[i] - s1 - s2) / A[i, i]
            if np.linalg.norm(x - x_old) < self.tolerance: return x
        raise RuntimeError("No convergió")


class InverseSolver(sistema_ecuaciones):
    """ 1. Inversa Directa (NumPy) """
    def solve(self, A, b):
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        try:
            inv = np.linalg.inv(A)
            return inv @ b
        except np.linalg.LinAlgError:
            raise ValueError("Matriz singular no tiene inversa")

class InverseGaussJordan(sistema_ecuaciones):
    """ 2. Inversa Gauss-Jordan (Manual) """
    def solve(self, A, b):
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        n = len(b)
        I = np.eye(n)
        Aug = np.column_stack((A, I))
        
        for k in range(n):
            max_index = np.argmax(np.abs(Aug[k:n, k])) + k
            if Aug[max_index, k] == 0: raise ValueError("Matriz singular")
            Aug[[k, max_index]] = Aug[[max_index, k]]
            Aug[k] = Aug[k] / Aug[k, k]
            for i in range(n):
                if i != k:
                    Aug[i] -= Aug[i, k] * Aug[k]
        
        A_inv = Aug[:, n:]
        return A_inv @ b


class TransformacionesLineales:
    def rotacion(self, vector, angulo_grados):
        theta = np.radians(angulo_grados)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        return R @ vector
    def escalacion(self, vector, sx, sy):
        S = np.array(((sx, 0), (0, sy)))
        return S @ vector
    def cizallamiento(self, vector, kx, ky):
        Sh = np.array(((1, kx), (ky, 1)))
        return Sh @ vector

class PageRankSolver:
    def calcular(self, M, d=0.85, max_iter=100, tol=1e-6):
        M = np.array(M, dtype=float)
        n = M.shape[0]
        sums = M.sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            M = np.where(sums != 0, M / sums, 0)
        v = np.ones(n) / n
        for _ in range(max_iter):
            v_new = d * (M @ v) + (1 - d) / n
            if np.linalg.norm(v_new - v) < tol: return v_new
            v = v_new
        return v

class DerivacionNumerica:
    def evaluar(self, func_str, x_val):
        allowed = {"x": x_val, "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp, "log": np.log, "tan": np.tan, "sqrt": np.sqrt}
        return eval(func_str, {"__builtins__": None}, allowed)
    def derivar(self, metodo, func_str, x, h):
        if metodo == "Adelante":
            return (self.evaluar(func_str, x + h) - self.evaluar(func_str, x)) / h
        elif metodo == "Atrás":
            return (self.evaluar(func_str, x) - self.evaluar(func_str, x - h)) / h
        elif metodo == "Central":
            return (self.evaluar(func_str, x + h) - self.evaluar(func_str, x - h)) / (2 * h)


class EigenSolver:
    """
    Clase para calcular Valores y Vectores Propios.
    """
    def metodo_potencia(self, A, max_iter=100, tol=1e-6):
        """
        Encuentra el valor propio dominante y su vector asociado.
        """
        A = np.array(A, dtype=float)
        n = A.shape[0]
        if A.shape != (n, n): raise ValueError("La matriz debe ser cuadrada")
        
        # Vector inicial aleatorio
        v = np.random.rand(n)
        v = v / np.linalg.norm(v)
        
        eigenvalue = 0
        
        for _ in range(max_iter):
            # Multiplicar Av
            w = A @ v
            
            # El nuevo valor propio es la norma (aproximadamente)
            # Nota: Usamos el cociente de Rayleigh para más precisión: (v.T @ A @ v) / (v.T @ v)
            new_eigenvalue = np.dot(v, w)
            
            # Normalizar para el siguiente paso
            norm_w = np.linalg.norm(w)
            if norm_w == 0: break # Vector nulo
            v_new = w / norm_w
            
            # Chequear convergencia
            if np.abs(new_eigenvalue - eigenvalue) < tol:
                return new_eigenvalue, v_new
            
            v = v_new
            eigenvalue = new_eigenvalue
            
        return eigenvalue, v

    def numpy_eig(self, A):
        """
        Usa el método QR implícito de NumPy para hallar TODOS los valores.
        """
        A = np.array(A, dtype=float)
        valores, vectores = np.linalg.eig(A)
        return valores, vectores