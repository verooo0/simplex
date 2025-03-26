import numpy as np
from numpy.linalg import inv
Mariona_is_autista = True

def leer_datos(dir_problema):

    with open(dir_problema, 'r') as file:
        lineas = file.readlines()

    c, A, b = [], [], []
    estado = None
    for linea in lineas:
        linea = linea.strip()
        if not linea:
            continue
        if linea.startswith("c="):
            estado = "c"
            continue
        if linea.startswith("A="):
            estado = "A"
            continue
        if linea.startswith("b="):
            estado = "b"
            linea = linea.replace("b=", "")
        if estado is not None  :
            valores = list(map(float, linea.strip().split()))
        if estado == "c":
            c.extend(valores)
        elif estado == "A":
            A.append(valores)
        elif estado == "b":
            b.extend(valores)
    
    return np.array(A), np.array(b), np.array(c)


A,b,c =leer_datos('./Problemes/prob1.txt')

import numpy as np
from numpy.linalg import inv

import numpy as np
from numpy.linalg import inv, LinAlgError

class Simplex:
    def __init__(self, A, b, c):
        self.A_orig = A.copy()  # Guardar matriz original
        self.b_orig = b.copy()
        self.c_orig = c.copy()
        self.m, self.n = A.shape  # m restricciones, n variables originales

    def faseI(self):
        A = self.A_orig
        b = self.b_orig
        m, n = self.m, self.n

        # Matriz aumentada: [A | I] para variables artificiales
        A_aug = np.hstack((A, np.eye(m)))
        c_aug = np.zeros(n + m)
        c_aug[n:] = 1  # Coste 1 para variables artificiales

        # Variables básicas iniciales (artificiales)
        var_b = list(range(n, n + m))
        var_nb = list(range(n))

        # Configurar matrices Ab y An
        Ab = A_aug[:, var_b]
        An = A_aug[:, var_nb]

        # Vectores de costes
        cb = c_aug[var_b]
        cn = c_aug[var_nb]

        # Solución inicial
        try:
            xb = np.linalg.solve(Ab, b)
        except LinAlgError:
            xb = np.linalg.lstsq(Ab, b, rcond=None)[0]

        z = np.dot(cb, xb)

        # Ejecutar Fase I
        try:
            var_b_resultado = self.__simplex(cn, cb, An, Ab, xb, z, var_nb, var_b, fase_I=True)
        except Exception as e:
            return str(e)

        # Eliminar variables artificiales de la base
        var_b_final = []
        for idx in var_b_resultado:
            if idx < n:  # Conservar solo variables originales
                var_b_final.append(idx)
            else:
                # Intentar pivotear fuera variables artificiales
                fila = var_b_resultado.index(idx)
                for j in var_nb:
                    if j < n and An[fila, j] != 0:
                        var_b_resultado[fila] = j
                        var_b_final.append(j)
                        break
                else:
                    raise Exception("Problema infactible")

        if len(var_b_final) != m:
            raise Exception("No se pudo eliminar todas las variables artificiales")

        return var_b_final

    def faseII(self, var_b):
        A = self.A_orig  # Usar matriz original
        b = self.b_orig
        c = self.c_orig
        m, n = self.m, self.n

        var_nb = [i for i in range(n) if i not in var_b]
        Ab = A[:, var_b]
        An = A[:, var_nb]

        cb = c[var_b]
        cn = c[var_nb]

        try:
            xb = np.linalg.solve(Ab, b)
        except LinAlgError:
            xb = np.linalg.lstsq(Ab, b, rcond=None)[0]

        z = np.dot(cb, xb)

        return self.__simplex(cn, cb, An, Ab, xb, z, var_nb, var_b, fase_I=False)

    def __simplex(self, cn, cb, An, Ab, xb, z, var_nb, var_b, fase_I):
        iter_max = 1000  # Prevenir bucles infinitos
        tol = 1e-10      # Tolerancia numérica

        for _ in range(iter_max):
            try:
                Ab_inv = inv(Ab)
            except LinAlgError:
                raise Exception("Matriz básica singular")

            # Costes reducidos
            r = cn - (cb @ Ab_inv) @ An

            if np.all(r >= -tol):
                if fase_I:
                    return var_b
                else:
                    return {"z": z, "xb": xb, "var_b": var_b}

            # Seleccionar variable entrante (más negativa)
            q = np.argmin(r)

            # Dirección de movimiento
            d = -Ab_inv @ An[:, q]

            if np.all(d >= -tol):
                raise Exception("Problema no acotado")

            # Calcular theta (máximo paso posible)
            theta_list = []
            for i in range(len(d)):
                if d[i] < -tol:
                    theta_list.append(-xb[i] / d[i])
                else:
                    theta_list.append(np.inf)
            theta = min(theta_list)
            p = theta_list.index(theta)

            # Actualizar variables básicas
            xb = xb + theta * d
            xb[p] = theta  # Asegurar precisión numérica

            # Actualizar valor de z
            z += theta * r[q]

            # Intercambiar variables básicas y no básicas
            var_b[p], var_nb[q] = var_nb[q], var_b[p]

            # Intercambiar columnas en An y Ab
            An[:, q], Ab[:, p] = Ab[:, p].copy(), An[:, q].copy()

            # Actualizar vectores de coste
            cb[p], cn[q] = cn[q], cb[p]

        raise Exception("Número máximo de iteraciones alcanzado")

    def solve(self):
        try:
            base_factible = self.faseI()
            resultado = self.faseII(base_factible)
            return resultado
        except Exception as e:
            return str(e)


simplex = Simplex(A, b, c)
x = simplex.solve()

print(x['z'], x['var_b'])

print((np.array(x['var_b']) + 1).tolist())

y ={2,7,8,5,10,1,16,18,13,12}

print(y == set((np.array(x['var_b']) + 1).tolist()))

