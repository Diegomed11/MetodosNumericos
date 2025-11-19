import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import metodosSolEcu as backend

class MathApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Suite Matemática Numérica Completa")
        self.root.geometry("750x600")

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill='both', padx=5, pady=5)

        self.tab_sistemas = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_sistemas, text='Sist. Ecuaciones')
        self.setup_sistemas()

        self.tab_transform = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_transform, text='Transformaciones 2D')
        self.setup_transform()

        self.tab_pagerank = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_pagerank, text='PageRank')
        self.setup_pagerank()
        
        self.tab_derivada = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_derivada, text='Derivación Num.')
        self.setup_derivada()
        
        
        self.tab_eigen = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_eigen, text='Valores Propios')
        self.setup_eigen()


    def setup_sistemas(self):
        frame = self.tab_sistemas
        
        ttk.Label(frame, text="Matriz A:").grid(row=0, column=0, sticky="w", padx=5)
        self.sys_text_A = tk.Text(frame, height=5, width=35)
        self.sys_text_A.grid(row=1, column=0, padx=5)
        # Matriz ejemplo simétrica para probar Cholesky
        self.sys_text_A.insert(tk.END, "4 1 1\n1 3 1\n1 1 2")

        ttk.Label(frame, text="Vector b:").grid(row=2, column=0, sticky="w", padx=5)
        self.sys_text_b = tk.Text(frame, height=5, width=35)
        self.sys_text_b.grid(row=3, column=0, padx=5)
        self.sys_text_b.insert(tk.END, "6\n5\n4")

        ttk.Label(frame, text="Seleccione Método:").grid(row=0, column=1, sticky="w")
        
       
        mis_metodos = [
            "--- Eliminación (3 Variantes) ---",
            "Gauss: Pivoteo Parcial (Estándar)",
            "Gauss: Gauss-Jordan",
            "Gauss: Pivoteo Total",
            "--- Factorización (4 Variantes) ---",
            "Factorización LU (Doolittle)",
            "Factorización LU (Crout)",
            "Factorización Cholesky (Simétrica)",
            "Factorización QR",
            "--- Iterativos ---",
            "Jacobi",
            "Gauss-Seidel",
            "--- Inversa (2 Variantes) ---",
            "Inversa: NumPy Directo",
            "Inversa: Gauss-Jordan Manual"
        ]
        
        self.sys_combo = ttk.Combobox(frame, values=mis_metodos, state="readonly", width=35)
        self.sys_combo.grid(row=1, column=1, sticky="ew", padx=5)
        self.sys_combo.current(1) # Seleccionar el primero válido

        ttk.Button(frame, text="Resolver Sistema", command=self.run_sistema).grid(row=2, column=1, sticky="ew", padx=5)

        ttk.Label(frame, text="Resultado x:").grid(row=3, column=1, sticky="w")
        self.sys_res = tk.Text(frame, height=5, width=35, state="disabled", bg="#f0f0f0")
        self.sys_res.grid(row=4, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

    def run_sistema(self):
        try:
            A = np.array([list(map(float, row.split())) for row in self.sys_text_A.get("1.0", tk.END).strip().splitlines()])
            b = np.array([float(val) for val in self.sys_text_b.get("1.0", tk.END).strip().splitlines()])
            
            metodo = self.sys_combo.get()
            solver = None
            
            # --- SELECTOR DE MÉTODOS ---
            if "---" in metodo:
                self.show_error(self.sys_res, "Selecciona un método válido, no un separador.")
                return
                
            # Eliminación
            if metodo == "Gauss: Pivoteo Parcial (Estándar)": solver = backend.GaussianElimination()
            elif metodo == "Gauss: Gauss-Jordan": solver = backend.GaussJordanElimination()
            elif metodo == "Gauss: Pivoteo Total": solver = backend.GaussianTotalPivoting()
            
            # Factorización (4 Métodos)
            elif metodo == "Factorización LU (Doolittle)": solver = backend.LUDoolittle()
            elif metodo == "Factorización LU (Crout)": solver = backend.LUCrout()
            elif metodo == "Factorización Cholesky (Simétrica)": solver = backend.CholeskyFactorization()
            elif metodo == "Factorización QR": solver = backend.QRFactorization()
            
            # Iterativos
            elif metodo == "Jacobi": solver = backend.Jacobi()
            elif metodo == "Gauss-Seidel": solver = backend.GaussSeidel()
            
            # Inversa (2 Métodos)
            elif metodo == "Inversa: NumPy Directo": solver = backend.InverseSolver()
            elif metodo == "Inversa: Gauss-Jordan Manual": solver = backend.InverseGaussJordan()

            if solver:
                res = solver.solve(A, b)
                self.show_output(self.sys_res, res)
            else:
                self.show_error(self.sys_res, "Método no reconocido")
                
        except Exception as e:
            self.show_error(self.sys_res, e)


    def setup_transform(self):
        frame = self.tab_transform
        ttk.Label(frame, text="Vector v (x, y):").grid(row=0, column=0, sticky="w", padx=5)
        self.trans_vec = ttk.Entry(frame)
        self.trans_vec.grid(row=1, column=0, padx=5); self.trans_vec.insert(0, "1, 1")
        ttk.Label(frame, text="Tipo:").grid(row=2, column=0, sticky="w", padx=5)
        self.trans_combo = ttk.Combobox(frame, values=["Rotación", "Escalación", "Cizallamiento"], state="readonly")
        self.trans_combo.grid(row=3, column=0, padx=5); self.trans_combo.current(0)
        self.param_frame = ttk.LabelFrame(frame, text="Parámetros")
        self.param_frame.grid(row=0, column=1, rowspan=4, padx=10, sticky="nsew")
        ttk.Label(self.param_frame, text="Ángulo:").grid(row=0, column=0); self.trans_angle = ttk.Entry(self.param_frame, width=8); self.trans_angle.grid(row=0, column=1)
        ttk.Label(self.param_frame, text="X:").grid(row=1, column=0); self.trans_x = ttk.Entry(self.param_frame, width=8); self.trans_x.grid(row=1, column=1)
        ttk.Label(self.param_frame, text="Y:").grid(row=2, column=0); self.trans_y = ttk.Entry(self.param_frame, width=8); self.trans_y.grid(row=2, column=1)
        ttk.Button(frame, text="Transformar", command=self.run_transform).grid(row=4, column=0, columnspan=2, pady=10)
        self.trans_res = tk.Text(frame, height=4, width=40, state="disabled", bg="#f0f0f0")
        self.trans_res.grid(row=5, column=0, columnspan=2, padx=5)

    def run_transform(self):
        try:
            vec = np.array([float(x) for x in self.trans_vec.get().replace(" ", "").split(",")])
            tipo = self.trans_combo.get()
            tool = backend.TransformacionesLineales()
            res = None
            if tipo == "Rotación": res = tool.rotacion(vec, float(self.trans_angle.get()))
            elif tipo == "Escalación": res = tool.escalacion(vec, float(self.trans_x.get()), float(self.trans_y.get()))
            elif tipo == "Cizallamiento": res = tool.cizallamiento(vec, float(self.trans_x.get()), float(self.trans_y.get()))
            self.show_output(self.trans_res, res)
        except Exception as e: self.show_error(self.trans_res, e)

    def setup_pagerank(self):
        frame = self.tab_pagerank
        ttk.Label(frame, text="Matriz Adyacencia:").pack(anchor="w", padx=5)
        self.page_matrix = tk.Text(frame, height=5, width=40); self.page_matrix.pack(padx=5); self.page_matrix.insert(tk.END, "0 1 1\n1 0 1\n0 0 1")
        ttk.Label(frame, text="Factor d:").pack(anchor="w", padx=5); self.page_d = ttk.Entry(frame); self.page_d.pack(padx=5); self.page_d.insert(0, "0.85")
        ttk.Button(frame, text="Calcular", command=self.run_pagerank).pack(pady=10)
        self.page_res = tk.Text(frame, height=5, width=40, state="disabled", bg="#f0f0f0"); self.page_res.pack(padx=5)

    def run_pagerank(self):
        try:
            M = np.array([list(map(float, r.split())) for r in self.page_matrix.get("1.0", tk.END).strip().splitlines()])
            res = backend.PageRankSolver().calcular(M, float(self.page_d.get()))
            self.show_output(self.page_res, res)
        except Exception as e: self.show_error(self.page_res, e)

    def setup_derivada(self):
        frame = self.tab_derivada
        ttk.Label(frame, text="Función f(x):").pack(anchor="w", padx=5); self.der_func = ttk.Entry(frame, width=30); self.der_func.pack(padx=5); self.der_func.insert(0, "x**2")
        fi = ttk.Frame(frame); fi.pack(pady=5)
        ttk.Label(fi, text="x:").grid(row=0, column=0); self.der_x = ttk.Entry(fi, width=8); self.der_x.grid(row=0, column=1); self.der_x.insert(0, "2.0")
        ttk.Label(fi, text="h:").grid(row=0, column=2); self.der_h = ttk.Entry(fi, width=8); self.der_h.grid(row=0, column=3); self.der_h.insert(0, "0.01")
        self.der_combo = ttk.Combobox(frame, values=["Adelante", "Atrás", "Central"], state="readonly"); self.der_combo.pack(padx=5); self.der_combo.current(2)
        ttk.Button(frame, text="Derivar", command=self.run_derivada).pack(pady=10)
        self.der_res = tk.Text(frame, height=4, width=40, state="disabled", bg="#f0f0f0"); self.der_res.pack(padx=5)

    def run_derivada(self):
        try:
            res = backend.DerivacionNumerica().derivar(self.der_combo.get(), self.der_func.get(), float(self.der_x.get()), float(self.der_h.get()))
            self.show_output(self.der_res, f"{res:.6f}")
        except Exception as e: self.show_error(self.der_res, e)
    
    def setup_eigen(self):
        frame = self.tab_eigen
        
        ttk.Label(frame, text="Matriz Cuadrada A:").pack(anchor="w", padx=5)
        self.eig_matrix = tk.Text(frame, height=6, width=40)
        self.eig_matrix.pack(padx=5, pady=5)
        # Matriz de ejemplo (simétrica para asegurar reales)
        self.eig_matrix.insert(tk.END, "2 1\n1 2") 

        ttk.Label(frame, text="Método:").pack(anchor="w", padx=5)
        self.eig_combo = ttk.Combobox(frame, values=["Método de la Potencia (Dominante)", "NumPy (Todos)"], state="readonly", width=30)
        self.eig_combo.pack(padx=5)
        self.eig_combo.current(0)

        ttk.Button(frame, text="Calcular Eigenvalores", command=self.run_eigen).pack(pady=10)
        
        ttk.Label(frame, text="Resultados:").pack(anchor="w", padx=5)
        self.eig_res = tk.Text(frame, height=8, width=50, state="disabled", bg="#f0f0f0")
        self.eig_res.pack(padx=5, pady=5)
    def run_eigen(self):
        try:
            # Leer matriz
            A = np.array([list(map(float, row.split())) for row in self.eig_matrix.get("1.0", tk.END).strip().splitlines()])
            
            metodo = self.eig_combo.get()
            tool = backend.EigenSolver()
            
            self.eig_res.config(state="normal", fg="black")
            self.eig_res.delete("1.0", tk.END)
            
            if "Potencia" in metodo:
                val, vec = tool.metodo_potencia(A)
                self.eig_res.insert(tk.END, f"Valor Propio Dominante:\n{val:.6f}\n\n")
                self.eig_res.insert(tk.END, f"Vector Propio Asociado:\n{np.array2string(vec, precision=4)}")
            else:
                vals, vecs = tool.numpy_eig(A)
                self.eig_res.insert(tk.END, "Valores Propios:\n")
                self.eig_res.insert(tk.END, np.array2string(vals, precision=4) + "\n\n")
                self.eig_res.insert(tk.END, "Vectores Propios (columnas):\n")
                self.eig_res.insert(tk.END, np.array2string(vecs, precision=4))
                
            self.eig_res.config(state="disabled")
            
        except Exception as e:
            self.show_error(self.eig_res, e)

    def show_output(self, widget, data):
        widget.config(state="normal", fg="black"); widget.delete("1.0", tk.END)
        if isinstance(data, np.ndarray): widget.insert(tk.END, np.array2string(data, precision=4, separator=', '))
        else: widget.insert(tk.END, str(data))
        widget.config(state="disabled")

    def show_error(self, widget, msg):
        widget.config(state="normal", fg="red"); widget.delete("1.0", tk.END); widget.insert(tk.END, f"Error: {msg}"); widget.config(state="disabled")

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style(); style.theme_use('clam') 
    app = MathApp(root)
    root.mainloop()