import numpy as np
from numpy.linalg import inv, LinAlgError

def llegir_dades(dir_problema):

    with open(dir_problema, 'r') as file:
        lineas = file.readlines()

    c, A, b = [], [], []
    estat = None
    for linea in lineas:
        linea = linea.strip()
        if not linea:
            continue
        if linea.startswith("c="):
            estat = "c"
            continue
        if linea.startswith("A="):
            estat = "A"
            continue
        if linea.startswith("b="):
            estat = "b"
            linea = linea.replace("b=", "")
        if estat is not None:
            valors = list(map(float, linea.strip().split()))
        if estat == "c":
            c.extend(valors)
        elif estat == "A":
            A.append(valors)
        elif estat == "b":
            b.extend(valors)
    
    return np.array(A), np.array(b), np.array(c)

class Simplex:
    def __init__(self, A, b, c, display=True):
        if display:
            print('Inici simplex primal amb regla de Bland')

        self.A_orig = A.copy()  # Guardar matriu original
        self.b_orig = b.copy()
        self.c_orig = c.copy()
        self.m, self.n = A.shape  # m restriccions, n variables originals
        self.iter = 1
        self.display = display

    def faseI(self):
        if self.display:
            print('Fase I')

        A = self.A_orig
        b = self.b_orig
        m, n = self.m, self.n

        # Matriu augmentada: [A | I] per a variables artificials
        A_aug = np.hstack((A, np.eye(m)))
        c_aug = np.zeros(n + m)
        c_aug[n:] = 1  # Cost 1 per a variables artificials

        # Variables bàsiques inicials (artificials)
        var_b = list(range(n, n + m))
        var_nb = list(range(n))

        # Configurar matrius Ab i An
        Ab = A_aug[:, var_b]
        An = A_aug[:, var_nb]

        # Vectors de costos
        cb = c_aug[var_b]
        cn = c_aug[var_nb]

        # Solució inicial
        xb = b.copy()  # Ja que Ab és la matriu identitat
        z = np.dot(cb, xb)

        # Executar Fase I
        var_b_resultado = self.__simplex(cn, cb, An, Ab, xb, z, var_nb, var_b, fase_I=True)

        # Eliminar variables artificials de la base
        var_b_final = []
        for idx in var_b_resultado:
            if idx < n:  # Conservar només variables originals
                var_b_final.append(idx)
            else:
                # Intentar pivotar fora variables artificials
                fila = var_b_resultado.index(idx)
                for j in var_nb:
                    if j < n and An[fila, j] != 0:
                        var_b_resultado[fila] = j
                        var_b_final.append(j)
                        break
                else:
                    raise Exception("Problema no factible")

        if len(var_b_final) != m:
            raise Exception("No s'han pogut eliminar totes les variables artificials")

        if self.display:
            print("Solució bàsica factible trobada, iteració", self.iter-1)

        return var_b_final

    def faseII(self, var_b):
        # Inicia la Fase II del mètode simplex per trobar la solució òptima
        if self.display:
            print('Fase II')

        # Recupera les matrius i vectors originals
        A = self.A_orig
        b = self.b_orig
        c = self.c_orig
        n = self.n

        # Determina les variables no bàsiques (les que no estan a la base actual)
        var_nb = [i for i in range(n) if i not in var_b]

        # Matriu de les variables bàsiques (Ab) i no bàsiques (An)
        Ab = A[:, var_b]
        An = A[:, var_nb]

        # Vectors de costos associats a les variables bàsiques (cb) i no bàsiques (cn)
        cb = c[var_b]
        cn = c[var_nb]

        # Calcula la solució inicial per a les variables bàsiques
        xb = np.linalg.solve(Ab, b)  
        z = np.dot(cb, xb)  # Calcula el valor inicial de la funció objectiu

        # Executa el mètode simplex per trobar la solució òptima
        resultat_faseII = self.__simplex(cn, cb, An, Ab, xb, z, var_nb, var_b, fase_I=False)

        if self.display:
            print(f'Solució òptima trobada, iteració {self.iter-1}, z = {resultat_faseII["z"]:.3f}')
        
        return resultat_faseII

    def __simplex(self, cn, cb, An, Ab, xb, z, var_nb, var_b, fase_I):
        iter_max = 1000  # Prevenir bucles infinits
        tol = 1e-10      # Tolerància numèrica

        for _ in range(iter_max):
            try:
                Ab_inv = inv(Ab)
            except LinAlgError:
                raise Exception("Matriu bàsica singular")

            # Costos reduïts
            r = cn - (cb @ Ab_inv) @ An

            if np.all(r >= -tol):
                if fase_I:               
                    if abs(z) > tol:
                        raise Exception("Problema no factible")
                    else:
                        return var_b
                else:
                    return {"z": z, "xb": xb, "var_b": var_b, "r": r}   

            # Seleccionar variable entrant (més negativa)
            
            # REGLA DE BLAND
            negative_indices = np.where(r < -tol)[0]
            if len(negative_indices) > 0:
                q = negative_indices[0]

            # Direcció de moviment
            d = -Ab_inv @ An[:, q]

            if np.all(d >= -tol):
                raise Exception("Problema no acotat")

            # Calcular theta (màxim pas possible)
            # REGLA DE BLAND
            theta = np.inf
            p = -1
            for i in range(len(d)):
                if d[i] < -tol:
                    theta_i = -xb[i] / d[i]
                    # Si és menor o hi ha empat però amb índex més petit
                    if theta_i < theta - tol or (abs(theta_i - theta) < tol and var_b[i] < var_b[p]):
                        theta = theta_i
                        p = i

            # Actualitzar variables bàsiques
            xb = xb + theta * d
            xb[p] = theta  # Assegurar precisió numèrica
            
            if np.all(np.abs(xb) < tol):
                raise Exception("Problema no factible")

            # Actualitzar valor de z
            z += theta * r[q]

            # Intercanviar variables bàsiques i no bàsiques
            var_b[p], var_nb[q] = var_nb[q], var_b[p]

            # Intercanviar columnes An i Ab
            An[:, q], Ab[:, p] = Ab[:, p].copy(), An[:, q].copy()

            # Actualitzar vectors de cost
            cb[p], cn[q] = cn[q], cb[p]

            if self.display:
                print(f"Iteració {self.iter}: iout = {p}, q = {q}, theta* = {theta:.3f}, z = {z:.3f}")
            self.iter += 1

        raise Exception("Nombre màxim d'iteracions assolit")

    def solve(self):
        try:
            base_factible = self.faseI()
            resultat_dict = self.faseII(base_factible)
            resultat_dict["var_b"] = set(np.array(resultat_dict["var_b"]) + 1)
            
            if self.display:
                print('Fi simplex primal')
                print('\nSolució òptima:\n')
                print(f'vb = {" ".join(map(str, resultat_dict["var_b"]))}')
                print(f'xb = {" ".join(map(lambda x: f"{x:.1f}", resultat_dict["xb"]))}')
                print(f'z = {resultat_dict["z"]:.3f}')
                print(f'r = {" ".join(map(lambda x: f"{x:.1f}", resultat_dict["r"]))}\n')

            return resultat_dict
        except Exception as e:
            if self.display:
                print(str(e))
            return str(e)


A, b, c = llegir_dades(f'./Problemes/prob1.txt')
simplex = Simplex(A, b, c, display=True)
resultat = simplex.solve()

for i in range(1, 9):
    print(f"Resultat per a prob{i}.txt:")
    A, b, c = llegir_dades(f'./Problemes/prob{i}.txt')
    simplex = Simplex(A, b, c, display=False)
    resultat = simplex.solve()
    if isinstance(resultat, str):
        print(f"Error: {resultat}")
    else:
        print(f'vb = {" ".join(map(str, resultat["var_b"]))}')
        print(f'xb = {" ".join(map(lambda x: f"{x:.1f}", resultat["xb"]))}')
        print(f'z = {resultat["z"]:.3f}')
        print(f'r = {" ".join(map(lambda x: f"{x:.1f}", resultat["r"]))}')
    print("-" * 50)


