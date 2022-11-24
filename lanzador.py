from Perceptron import Perceptron
from Perceptron_con_tensorflow import PerceptronTensorFlow
import helpers
from numpy import array

def menu():
    print("========================")
    print(" BIENVENIDO A PREDICCIÓN_NEURONAS ")
    print("========================")
    print("[1] Perceptron  ")
    print("[2] Perceptron con tensorflow ")
    print("[3] Salir ")
    print("========================")

def lanzar():
    Perceptron1=Perceptron(array([[1, 0],[1, 1],[0, 1],[0, 0]]),array([[0],[1], [0],[0]]),-1,1,1,0,0.1,3000)
    #PerceptronTensorFlow1=PerceptronTensorFlow([[1., 0.], [1., 1.], [0., 1.], [0., 0.]],[[0.], [1.], [0.], [0.]],10000)
    while True:
        menu()
        opcion=int(input("> "))
        helpers.limpiar_pantalla()
        if opcion == 1:
            Perceptron1.aprendizaje()
            Perceptron1.visualización()

        if opcion == 2:
            pass
            #PerceptronTensorFlow1.aprendizaje()
            #PerceptronTensorFlow1.visualización()

        if opcion == 3:
            print("Saliendo...\n")
            break