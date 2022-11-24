from numpy import exp, array, random
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self,observaciones_entradas,predicciones,limiteMin,limiteMax,sesgo,wb,txAprendizaje,epochs):
        self.observaciones_entradas = observaciones_entradas
        self.predicciones = predicciones
        self.limiteMin = limiteMin
        self.limiteMax = limiteMax
        random.seed(1)
        self.w11=(limiteMax-limiteMin) * random.random() + limiteMin
        self.w21=(limiteMax-limiteMin) * random.random() + limiteMin
        self.w31=(limiteMax-limiteMin) * random.random() + limiteMin
        self.sesgo=sesgo
        self.wb=wb
        self.txAprendizaje=txAprendizaje
        self.peso=[self.w11,self.w21,self.w31,self.wb]
        self.epochs=epochs
        self.Grafica_MSE=[]

    def suma_ponderada(self,X1,W11,X2,W21,B,WB):
        return (B*WB+( X1*W11 + X2*W21))

    def funcion_activacion_sigmoide(self,valor_suma_ponderada):
        return (1 / (1 + exp(-valor_suma_ponderada)))

    def funcion_activacion_relu(self,valor_suma_ponderada):
        return (max(0,valor_suma_ponderada))

    def error_lineal(self,valor_esperado, valor_predicho):
        return (valor_esperado-valor_predicho)

    def calculo_gradiente(self,valor_entrada,prediccion,error):
        return (-1 * error * prediccion * (1-prediccion) * valor_entrada)

    def calculo_valor_ajuste(self,valor_gradiente, tasa_aprendizaje):
        return (valor_gradiente*tasa_aprendizaje)

    def calculo_nuevo_peso (self,valor_peso, valor_ajuste):
        return (valor_peso - valor_ajuste)

    def calculo_MSE(self,predicciones_realizadas, predicciones_esperadas):
        i=0;
        suma=0;
        for prediccion in predicciones_esperadas:
            diferencia = predicciones_esperadas[i] - predicciones_realizadas[i]
            cuadradoDiferencia = diferencia * diferencia
            suma = suma + cuadradoDiferencia
        media_cuadratica = 1 / (len(predicciones_esperadas)) * suma
        return media_cuadratica

    def aprendizaje(self):
        for epoch in range(0,self.epochs):
            print("EPOCH ("+str(epoch)+"/"+str(self.epochs)+")")
            predicciones_realizadas_durante_epoch = [];
            predicciones_esperadas = [];
            numObservacion = 0
            for observacion in self.observaciones_entradas:

                #Carga de la capa de entrada
                x1 = observacion[0];
                x2 = observacion[1];

                #Valor de predicción esperado
                valor_esperado = self.predicciones[numObservacion][0]

                #Etapa 1: Cálculo de la suma ponderada
                valor_suma_ponderada = self.suma_ponderada(x1,self.w11,x2,self.w21,self.sesgo,self.wb)


                #Etapa 2: Aplicación de la función de activación
                valor_predicho = self.funcion_activacion_sigmoide(valor_suma_ponderada)


                #Etapa 3: Cálculo del error
                valor_error = self.error_lineal(valor_esperado,valor_predicho)


                #Actualización del peso 1
                #Cálculo ddel gradiente del valor de ajuste y del peso nuevo
                gradiente_W11 = self.calculo_gradiente(x1,valor_predicho,valor_error)
                valor_ajuste_W11 = self.calculo_valor_ajuste(gradiente_W11,self.txAprendizaje)
                self.w11 = self.calculo_nuevo_peso(self.w11,valor_ajuste_W11)

                # Actualización del peso 2
                gradiente_W21 = self.calculo_gradiente(x2, valor_predicho, valor_error)
                valor_ajuste_W21 = self.calculo_valor_ajuste(gradiente_W21, self.txAprendizaje)
                self.w21 = self.calculo_nuevo_peso(self.w21, valor_ajuste_W21)


                # Actualización del peso del sesgo
                gradiente_Wb = self.calculo_gradiente(self.sesgo, valor_predicho, valor_error)
                valor_ajuste_Wb =self. calculo_valor_ajuste(gradiente_Wb, self.txAprendizaje)
                self.wb = self.calculo_nuevo_peso(self.wb, valor_ajuste_Wb)

                print("     EPOCH (" + str(epoch) + "/" + str(self.epochs) + ") -  Observación: " + str(numObservacion+1) + "/" + str(len(self.observaciones_entradas)))

                #Almacenamiento de la predicción realizada:
                predicciones_realizadas_durante_epoch.append(valor_predicho)
                predicciones_esperadas.append(self.predicciones[numObservacion][0])

                #Paso a la observación siguiente
                numObservacion = numObservacion+1

            MSE = self.calculo_MSE(predicciones_realizadas_durante_epoch, self.predicciones)
            self.Grafica_MSE.append(MSE[0])
            print("MSE: "+str(MSE))



    def visualización(self):
        plt.plot(self.Grafica_MSE)
        plt.ylabel('MSE')
        plt.show()


        print()
        print()
        print ("¡Aprendizaje terminado!")
        print ("Pesos iniciales: " )
        print ("W11 = "+str(self.peso[0]))
        print ("W21 = "+str(self.peso[1]))
        print ("Wb = "+str(self.peso[3]))

        print ("Pesos finales: " )
        print ("W11 = "+str(self.w11))
        print ("W21 = "+str(self.w21))
        print ("Wb = "+str(self.wb))

        print()
        print("--------------------------")
        print ("PREDICCIÓN ")
        print("--------------------------")
        x1 = 1
        x2 = 1

        #Etapa 1: Cálculo de la suma ponderada
        valor_suma_ponderada = self.suma_ponderada(x1,self.w11,x2,self.w21,self.sesgo,self.wb)


        #Etapa 2: Aplicación de la función de activación
        valor_predicho = self.funcion_activacion_sigmoide(valor_suma_ponderada)
        #valor_predicho = funcion_activacion_relu(valor_suma_ponderada)

        print("Predicción del [" + str(x1) + "," + str(x2)  + "]")
        print("Predicción = " + str(valor_predicho))

