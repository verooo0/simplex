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

class Simplex:

    def __init__(self, A, b, c):
        self.A = A
        self.b = b
        self.c = c

    def faseI(self):
        A = self.A
        b = self.b
        c = self.c

        c_f = [0] * len(c) + [1] * len(A)
        c_f = np.array(c_f)
        var_b = np.array([i for i in range(len(c),len(c)+len(A)) ])
        var_nb = np.array([i for i in range(len(c))])
        Ab =np.eye(len(A))
        An = A
        cn= np.array([0] * len(c))
        cb = np.array([1] * len(A))
        xb = np.linalg.solve(Ab, b) # Comprobar que faci la inversa
        #xn = np.array([0] * len(c))
        z = np.dot(cb,xb)
        #self.faseII(An,b,c)
        return self.__simplex(cn, cb, An, Ab, xb, z, var_nb, var_b)
    

    def faseII(self, var_b):
        A = self.A
        b = self.b
        c = self.c

        var_nb = np.array([i for i in range(len(c)) if i not in var_b])
        
        Ab = A[:, var_b]
        An = A[:, var_nb]

        cb = c[var_b]
        
        cn = c[var_nb]
 
        xb = np.dot(inv(Ab), b)

        z = np.dot(cb, xb)

        return self.__simplex(cn, cb, An, Ab, xb, z, var_nb, var_b)

        

    def __simplex(self, cn, cb, An, Ab, xb, z, var_nb, var_b):
        iteracio = 0

        while True:  # Corrected loop condition
            # Compute inverse of Ab
            Ab_inv = inv(Ab)
            # Calculate reduced costs
            r = cn - np.dot(cb, np.dot(Ab_inv, An))
            # Check for optimality
            if all(r >= 0):
                print(z)
                return var_b  # Optimal solution
            # Select entering variable (most negative reduced cost)
            q = np.argmin(r)
            # Calculate direction vector
            d = np.dot(-Ab_inv, An[:, q])
            # Check for unbounded problem
            if all(d >= 0):
                raise Exception("El problema es no acotado")
            # Compute theta and exiting variable
            theta_lst = [-xb[i] / d[i] if d[i] < 0 else np.inf for i in range(len(d))]
            theta = min(theta_lst)
            p = theta_lst.index(theta)
            # Update basic variables
            xb = xb + theta * d
            xb[p] = theta  # Replace outgoing variable with theta
            # Update objective value
            z += theta * r[q]
            # Swap basic and non-basic variables
            var_b[p], var_nb[q] = var_nb[q], var_b[p]
            # Update matrices An and Ab
            An[:, q], Ab[:, p] = Ab[:, p].copy(), An[:, q].copy()  # Ensure deep copy
            # Update cost vectors
            cb[p], cn[q] = cn[q], cb[p]

            iteracio += 1

            # Use q in subsequent logic (example placeholder below)
            print(f"Entering variable index q: {q}")



    def solve(self):
        solucio_inicial = self.faseI()
        print(solucio_inicial)
        resultat = self.faseII(solucio_inicial)
        print(resultat)
        return resultat


simplex = Simplex(A, b, c)
simplex.solve()