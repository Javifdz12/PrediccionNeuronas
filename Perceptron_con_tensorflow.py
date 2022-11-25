import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class PerceptronTensorFlow:
    def __init__(self,valores_entradas_X,valores_a_predecir_Y,epochs):
        self.valores_entradas_X = valores_entradas_X
        self.valores_a_predecir_Y=valores_a_predecir_Y
        self.epochs = epochs
        #Variable TensorFLow correspondiente a los valores de neuronas de entrada
        self.tf_neuronas_entradas_X = tf.placeholder(tf.float32, [None, 2])

        #Variable TensorFlow correspondiente a la neurona de salida (predicción real)
        self.tf_valores_reales_Y = tf.placeholder(tf.float32, [None, 1])

        #-- Peso --
        #Creación de una variable TensorFlow de tipo tabla
        #que contiene 2 entradas y cada una tiene un peso [2,1]
        #Estos valores se inicializan al azar
        self.peso = tf.Variable(tf.random_normal([2, 1]), tf.float32)

        #-- Sesgo inicializado a 0 --
        self.sesgo = tf.Variable(tf.zeros([1, 1]), tf.float32)

        #La suma ponderada es en la práctica una multiplicación de matrices
        #entre los valores en la entrada X y los distintos pesos
        #la función matmul se encarga de hacer esta multiplicación
        self.sumaponderada = tf.matmul(self.tf_neuronas_entradas_X,self.peso)

        #Adición del sesgo a la suma ponderada
        self.sumaponderada = tf.add(self.sumaponderada,self.sesgo)

        #Función de activación de tipo sigmoide que permite calcular la predicción
        self.prediccion = tf.sigmoid(self.sumaponderada)

        #Función de error de media cuadrática MSE
        self.funcion_error = tf.reduce_sum(tf.pow(self.tf_valores_reales_Y-self.prediccion,2))

        #Descenso de gradiente con una tasa de aprendizaje fijada a 0,1
        self.optimizador = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(self.funcion_error)
        self.init = tf.global_variables_initializer()
        self.sesion = tf.Session()
        self.Grafica_MSE=[]

    def aprendizaje(self):
        self.sesion.run(self.init)
        for i in range(self.epochs):

            #Realización del aprendizaje con actualzación de los pesos
            self.sesion.run(self.optimizador, feed_dict = {self.tf_neuronas_entradas_X: self.valores_entradas_X, self.tf_valores_reales_Y:self.valores_a_predecir_Y})

            #Calcular el error
            MSE = self.sesion.run(self.funcion_error, feed_dict = {self.tf_neuronas_entradas_X: self.valores_entradas_X, self.tf_valores_reales_Y:self.valores_a_predecir_Y})

            #Visualización de la información
            self.Grafica_MSE.append(MSE)
            print("EPOCH (" + str(i) + "/" + str(self.epochs) + ") -  MSE: "+ str(MSE))

    def visualización(self):
        plt.plot(self.Grafica_MSE)
        plt.ylabel('MSE')
        plt.show()

        print("--- VERIFICACIONES ----")

        for i in range(0,4):
            print("Observación:"+str(self.valores_entradas_X[i])+ " - Esperado: "+str(self.valores_a_predecir_Y[i])+" - Predicción: "+str(self.sesion.run(self.prediccion, feed_dict={self.tf_neuronas_entradas_X: [self.valores_entradas_X[i]]})))
        self.sesion.close()


