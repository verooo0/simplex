import numpy as np

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


def faseI(A, b, c):
    c_f = [0] * len(c) + [1] * len(A)
    c_f = np.array(c_f)
    var_b = [i for i in range(len(c),len(c)+len(A)) ]
    var_nb = [i for i in range(len(c))]
    Ab =np.eye(len(A))
    An = A
    cn= np.array([0] * len(c))
    cb = np.array([1] * len(A))
    xb = np.linalg.solve(Ab, b) # Comprobar que faci la inversa
    xn = np.array([0] * len(c))
    z = np.dot(cb,xb)
    faseII(An,b,c)
    print(Ab)

def faseII(A, b, c):
    pass

def simplex(A,b,c):
    pass

def solve(A,b,c):
    #fase 1
    faseI(A,b,c)

    faseII(A,b,c)
    pass

faseI(A,b,c)