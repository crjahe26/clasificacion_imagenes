### Reporte Técnico del Modelo

#### 1. Objetivo del Modelo:

El objetivo del modelo es clasificar imágenes de personas en dos categorías: con gafas y sin gafas. El modelo utiliza una arquitectura de red neuronal convolucional (CNN) y se entrena con un conjunto de datos que incluye variaciones de pose, expresión, ojos y tamaño.

#### 2. Descripción del Modelo:

El modelo se compone de varias capas convolucionales seguidas de capas de pooling para extraer y reducir características espaciales. Luego, se aplica una capa completamente conectada con una capa de dropout para prevenir el sobreajuste. Finalmente, se utiliza una capa de salida con activación softmax para la clasificación binaria.

#### 3. Preprocesamiento de Datos:

- Las imágenes se cargan y redimensionan a una resolución de 64x60 píxeles.
- Se aplica aumento de datos durante el entrenamiento para mejorar la generalización y la robustez del modelo.

#### 4. Arquitectura del Modelo:

- Capa de entrada: Imágenes en escala de grises redimensionadas a 64x60 píxeles.
- Capa convolucional (64 filtros, kernel size 3x3, función de activación ReLU).
- Capa de max pooling (reduce la dimensionalidad de la imagen).
- Capa convolucional (128 filtros, kernel size 3x3, función de activación ReLU).
- Capa de max pooling.
- Capa totalmente conectada (256 neuronas, función de activación ReLU).
- Capa de dropout (reduce el sobreajuste).
- Capa de salida (2 neuronas, función de activación softmax para clasificación binaria).

#### 5. Entrenamiento del Modelo:

- Se divide el conjunto de datos en conjuntos de entrenamiento y prueba (80% - 20%).
- Se utiliza la función de pérdida de entropía cruzada categórica y el optimizador Adam con una tasa de aprendizaje reducida a 0.0001.
- El modelo se entrena durante 100 épocas con un tamaño de lote de 64.

#### 6. Evaluación del Modelo:

- El modelo se evalúa en un conjunto de prueba independiente.
- La precisión del modelo en el conjunto de prueba es monitoreada y se imprime al final del entrenamiento.

#### 7. Resultados Iniciales:

- En la iteración inicial, la precisión en el conjunto de prueba es monitoreada y se imprime al final del entrenamiento.

#### 8. Almacenamiento del Modelo:

- El modelo se guarda en un archivo (formato H5) para permitir su reutilización en el futuro.


### Análisis y Conclusiones

#### Resultados de la Última Epoch (Epoch 100):

- **Pérdida de Entrenamiento (Loss):** 0.3614
- **Precisión de Entrenamiento (Accuracy):** 83.23%
- **Pérdida de Validación (Validation Loss):** 0.2152
- **Precisión de Validación (Validation Accuracy):** 90.13%

#### Matriz de Confusión:

|               | Predicción Gafas | Predicción Sin Gafas |
|---------------|------------------|----------------------|
| Real Gafas    |       0.82       |         0.18         |
| Real Sin Gafas|        0         |           1          |



#### Test Accuracy:

- La precisión en el conjunto de prueba es del 90.13%.

#### Conclusiones:

1. **Rendimiento General:**
   - El modelo ha demostrado un buen rendimiento con una precisión de validación del 90.13%.

2. **Matriz de Confusión:**
   - La matriz de confusión muestra que el modelo tiene un rendimiento sólido en ambas clases (gafas y sin gafas).
   - La mayoría de las predicciones se encuentran en la diagonal principal, indicando una clasificación precisa.

3. **Tendencia de Entrenamiento:**
   - La pérdida de entrenamiento disminuyó gradualmente a lo largo de las épocas, indicando un aprendizaje efectivo.
   - La precisión de entrenamiento también mejoró, llegando al 83.23%.

4. **Posible Sobreajuste:**
   - La diferencia entre la precisión de entrenamiento y la precisión de validación podría sugerir cierto sobreajuste, pero la brecha no es excesivamente grande.

5. **Ajuste Adicional:**
   - Se podría explorar la posibilidad de ajustar hiperparámetros como la tasa de aprendizaje, la tasa de dropout y la complejidad de la red para mejorar aún más el rendimiento.

6. **Conclusión General:**
   - El modelo ha alcanzado un nivel aceptable de precisión y generalización en la clasificación de personas con y sin gafas.