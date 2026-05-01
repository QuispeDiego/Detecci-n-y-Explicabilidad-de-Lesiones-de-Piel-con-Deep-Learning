# 🔬 Detección y Explicabilidad de Lesiones de Piel con Deep Learning

Clasificación de lesiones dermatológicas usando **ResNet50** con transfer learning, aplicando **LIME** para explicar las predicciones del modelo (Explainable AI / XAI).

---

## 📋 Descripción

Este proyecto entrena un modelo de deep learning para clasificar imágenes de lesiones de piel en 7 categorías clínicas, usando el dataset público **HAM10000**. Adicionalmente, se aplica la técnica de explicabilidad **LIME** para visualizar qué regiones de la imagen influyen en cada predicción — lo que permite entender el razonamiento del modelo.

**Precisión obtenida: 76%**

---

## 🏷️ Clases detectadas

| # | Lesión |
|---|--------|
| 0 | Melanocytic nevi |
| 1 | Melanoma |
| 2 | Benign keratosis-like lesions |
| 3 | Basal cell carcinoma |
| 4 | Actinic keratoses |
| 5 | Vascular lesions |
| 6 | Dermatofibroma |

---

## 🛠️ Tecnologías utilizadas

- **Python**
- **TensorFlow / Keras** — entrenamiento del modelo
- **ResNet50** — transfer learning con pesos de ImageNet
- **LIME** (`lime`) — explicabilidad del modelo (XAI)
- **Scikit-learn** — preprocesamiento y métricas
- **Pandas / NumPy** — manejo de datos
- **Matplotlib** — visualización

---

## 📂 Dataset

**HAM10000** (Human Against Machine with 10000 training images)  
Dataset público de imágenes dermatológicas con más de 10,000 imágenes etiquetadas.

🔗 [Ver dataset en Kaggle](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection)

---

## 🧠 Arquitectura del modelo

```
ResNet50 (preentrenado con ImageNet, sin capa top)
    → GlobalAveragePooling2D
    → Flatten
    → Dense(512, activation='relu')
    → Dropout(0.2)
    → Dense(7, activation='softmax')
```

- **Optimizador**: SGD (learning_rate=0.01)
- **Loss**: Categorical Crossentropy
- **Épocas**: 30
- **Input shape**: (299, 299, 3)

---

## 💡 Explicabilidad con LIME

LIME (Local Interpretable Model-Agnostic Explanations) permite visualizar qué zonas de la imagen el modelo considera relevantes para cada clasificación.

Para cada imagen se generan visualizaciones de las 7 clases mostrando:
- Regiones con influencia **positiva** en la predicción
- Regiones con influencia **negativa** en la predicción
