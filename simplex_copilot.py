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

import numpy as np
from numpy.linalg import inv

import numpy as np
from numpy.linalg import inv, LinAlgError

class Simplex:
    def __init__(self, A, b, c, display=True):
        if display:
            print('Inici simplex primal amb regla de Bland')

        self.A_orig = A.copy()  # Guardar matriz original
        self.b_orig = b.copy()
        self.c_orig = c.copy()
        self.m, self.n = A.shape  # m restricciones, n variables originales
        self.iter = 1
        self.display = display

    def faseI(self):
        if self.display:
            print('Fase I')

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

        if self.display:
            print("Solució bàsica factible trobada, iteració", self.iter-1)

        return var_b_final

    def faseII(self, var_b):
        if self.display:
            print('Fase II')

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

        resultat_faseII = self.__simplex(cn, cb, An, Ab, xb, z, var_nb, var_b, fase_I=False)

        if self.display:
            print(f'Solució òptima trobada, iteració {self.iter-1}, z = {resultat_faseII['z']:.3f}')
        return resultat_faseII

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
                    return {"z": z, "xb": xb, "var_b": var_b, "r": r}   

            # Seleccionar variable entrante (más negativa)

            # REGLA DE BLAND
            negative_indices = np.where(r < -tol)[0]
            if len(negative_indices) > 0:
                q = negative_indices[0]

            # Dirección de movimiento
            d = -Ab_inv @ An[:, q]

            if np.all(d >= -tol):
                raise Exception("Problema no acotado")

            # Calcular theta (máximo paso posible)
            # REGLA DE BLAND
            theta = np.inf
            p = -1
            for i in range(len(d)):
                if d[i] < -tol:
                    theta_i = -xb[i] / d[i]
                    # Si es menor o hay empate pero con índice más pequeño
                    if theta_i < theta - tol or (abs(theta_i - theta) < tol and var_b[i] < var_b[p]):
                        theta = theta_i
                        p = i

            # Actualizar variables básicas
            xb = xb + theta * d
            xb[p] = theta  # Asegurar precisión numérica
            
            if np.all(np.abs(xb) < tol):
                raise Exception("Problema no factible")

            # Actualizar valor de z
            z += theta * r[q]

            # Intercambiar variables básicas y no básicas
            var_b[p], var_nb[q] = var_nb[q], var_b[p]

            # Intercambiar columnas en An y Ab
            An[:, q], Ab[:, p] = Ab[:, p].copy(), An[:, q].copy()

            # Actualizar vectores de coste
            cb[p], cn[q] = cn[q], cb[p]

            if self.display:
                print(f"Iteració {self.iter}: iout = {p}, q = {q}, theta* = {theta:.3f}, z = {z:.3f}")
            self.iter += 1

        raise Exception("Número máximo de iteraciones alcanzado")

    def solve(self):
        try:
            base_factible = self.faseI()
            resultat_dict = self.faseII(base_factible)
            resultat = resultat_dict["z"], set(np.array(resultat_dict["var_b"]) + 1)
            
            if self.display:
                print('Fi simplex primal')
                print('\nSolució òptima:\n')
                print(f'vb = {' '.join(map(str, resultat_dict["var_b"]))}')
                print(f'xb = {' '.join(map(lambda x: f"{x:.1f}", resultat_dict["xb"]))}')
                print(f'z = {resultat_dict["z"]:.3f}')
                print(f'r = {' '.join(map(lambda x: f"{x:.1f}", resultat_dict["r"]))}\n')

            return resultat
        except Exception as e:
            return str(e)


A, b, c = leer_datos(f'./Problemes/prob1.txt')
simplex = Simplex(A, b, c, display=True)
resultat = simplex.solve()


"""for i in range(1, 9):
    print(f"Resultado para prob{i}.txt:")
    A, b, c = leer_datos(f'./Problemes/prob{i}.txt')
    simplex = Simplex(A, b, c, display=False)
    resultat = simplex.solve()
    print(resultat)
    print("-" * 50)

"""
