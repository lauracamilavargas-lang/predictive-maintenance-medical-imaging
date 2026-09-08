# Sistema de Mantenimiento Predictivo — Equipos de Imagenología Médica

Metodología de mantenimiento predictivo para equipos biomédicos de funcionamiento basado en emisión de radiación ionizante. La capa predictiva es un clasificador binario XGBoost (Modelo A) que demuestra que la recurrencia temprana de fallas es estadísticamente predecible; sobre esa base se construye una matriz histórica de priorización y, a partir de ella, los protocolos de mantenimiento.

Desarrollado en: Universidad Autónoma de Occidente · Ingeniería Biomédica
Período del dataset: 2020 – 2025
Entorno de ejecución: Google Colab

## Descripción general

El sistema analiza las órdenes de mantenimiento correctivo de siete tipos de equipos de imagenología (TAC, Angiografía, Fluoroscopía, RX Fijo, RX Móvil, Arco C y Mamografía) registradas entre 2020 y 2025. Tras el pre-procesamiento y el filtrado por trazabilidad temporal, el conjunto de modelado (`df_modelo`) queda con 1.199 registros válidos.

La pregunta que responde la capa predictiva es binaria: ante un evento de fallo, ¿el equipo volverá a fallar dentro de un horizonte de 30 días? Se clasifica cada evento como **Requiere acción** (próxima falla en ≤30 días) o **Estable** (después). El entregable central no es un modelo de predicción en tiempo real, sino una metodología que traduce el fenómeno de recurrencia temprana en listas de chequeo de mantenimiento basadas en evidencia histórica.

La metodología se organiza en tres capas separadas:

1. **Modelo A (XGBoost binario)** — demuestra que la recurrencia temprana es predecible.
2. **Matriz histórica de priorización** — localiza y prioriza sistemas y componentes por recurrencia observada, frecuencia y nivel de evidencia. No cruza predicciones individuales del modelo.
3. **Protocolo / checklist** — se construye a partir de la matriz, los manuales del fabricante y el criterio de ingeniería clínica; nunca directamente de las salidas del modelo.

## Resultados

Selección del Modelo A en dos etapas: comparación en conjunto de prueba y validación temporal.

**Etapa 1 — Conjunto de prueba (split aleatorio, F1-macro a umbral por defecto)**

| Modelo              | F1-macro |
|---------------------|----------|
| Baseline (Dummy)    | 0.389    |
| Regresión Logística | 0.647    |
| Random Forest       | 0.658    |
| XGBoost             | 0.651    |

Random Forest y XGBoost quedan en empate técnico (diferencia de 0.007, dentro del ruido de muestreo). La Regresión Logística por debajo de los no lineales es coherente con una frontera de decisión no linealmente separable.

**Etapa 2 — Validación temporal expansiva (walk-forward)**

| Métrica                              | XGBoost | Random Forest |
|--------------------------------------|---------|---------------|
| F1-macro                             | 0.616   | 0.625         |
| Recall clase "Requiere acción"       | 0.723   | 0.742         |
| Desviación de F1-macro entre folds   | 0.018   | 0.027         |

Ambos modelos son estadísticamente equivalentes, con Random Forest levemente por delante en F1-macro y recall (dentro del ruido). La validación cruzada estratificada de 5 folds confirma la estabilidad: F1-macro medio de 0.628 (desviación 0.042), a 0.023 del conjunto de prueba.

**Selección de XGBoost.** No se elige por rendir más —Random Forest incluso presenta un recall temporal ligeramente superior—, sino por mejor generalización (brecha train–test de +0.027 frente a +0.047 de Random Forest), mayor estabilidad entre periodos y mayor parsimonia (profundidad 2 frente a 5), características prioritarias para un modelo destinado a anticipar fallas futuras.

## Estructura del repositorio

```
predictive-maintenance-medical-imaging/
├── Modelo_para_protocolo_predictivo_TG.ipynb   # notebook principal
├── README.md
├── LICENSE
├── anexo_completo.html                         # notebook exportado (código + salidas)
├── matriz_priorizacion.csv                     # matriz histórica de priorización
├── matriz_priorizacion.html                    # versión navegable de la matriz
└── assets/                                      # figuras generadas (fondo blanco)
    ├── analisis_tbf.png
    ├── distribucion_target.png
    ├── recurrencia_por_tipo.png
    ├── resumen_por_tipo.png
    ├── features_numericas.png
    ├── separabilidad_umap.png
    ├── variables_modelo_a.png
    ├── comparativa_4_modelos_a.png
    ├── comparativa_4_modelos_clase_a.png
    ├── validacion_temporal_comparativa.png
    ├── desempeno_por_tipo.png
    └── sensibilidad_horizonte.png
```

El dataset del hospital no se incluye por restricciones de confidencialidad clínica.

## Arquitectura del pipeline

```
Excel del hospital
        │
        ▼
┌─────────────────────────────────────────┐
│  Bloque 1 — Pre-procesamiento           │
│  · Limpieza y estandarización           │
│  · Criticidad global (1-5) y temporal   │
│  · Features temporales anti-fuga        │
│  · df_modelo: 1.199 registros           │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│  Bloque 2 — EDA                         │
│  · Construcción del target (30 días)    │
│  · Distribución TBF y umbral            │
│  · Recurrencia por tipo                 │
│  · Separabilidad UMAP (silhouette)      │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│  Bloque 3 — Modelado y entrenamiento    │
│  · Baseline, LR, RF, XGBoost            │
│  · Modelo A (XGBoost binario)           │
│  · Validación temporal expansiva        │
│  · Desempeño por tipo · sensibilidad    │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│  Bloque 4 — Validación del modelo       │
│  · Validación cruzada 5-fold            │
│  · Experimento de balanceo de clases    │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│  Bloque 5 — Capa descriptiva            │
│  · Frecuencia por sistema               │
│  · Pareto de componentes                │
│  · Heatmap recurrencia tipo × sistema   │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│  Bloque 6 — Construcción de protocolos  │
│  · Matriz histórica de priorización     │
│  · Base para las listas de chequeo      │
└─────────────────────────────────────────┘
```

## Estructura del notebook

**Bloque 1 — Pre-procesamiento**
Carga y consolidación del Excel; estandarización de columnas; limpieza de texto y corrección de `sistema_general` (11 clases consolidadas); tratamiento de horas de parada; variable de cambio de componente; índice de criticidad 1–5 por tipo × sistema y su versión temporal past-only; features temporales por equipo (TBF, fallos en 30 y 90 días, edad, horas de parada acumuladas) y derivadas (TBF promedio histórico, tendencia, tasa de fallo, TBF anterior). Todas las ventanas usan solo información anterior a cada evento. Resultado: `df_modelo` con 1.199 registros.

**Bloque 2 — EDA**
Construcción del target `periodo_fallo` (binario, umbral de 30 días hacia adelante) y su trazabilidad; análisis de la distribución del TBF que fundamenta el umbral; verificación de censura por cierre del estudio; resumen por tipo de equipo; variables de entrada del Modelo A; distribución del target; patrón de recurrencia por tipo; caracterización de features numéricas; análisis bivariado; separabilidad del target por UMAP con coeficiente de silueta.

**Bloque 3 — Modelado y entrenamiento**
Baseline (`DummyClassifier`); comparadores lineal (Regresión Logística) y no lineales (Random Forest, CatBoost); entrenamiento del Modelo A (XGBoost binario) con ponderación de clases y ajuste del umbral de decisión sobre entrenamiento; comparativa de los cuatro modelos en igualdad de condiciones; validación temporal expansiva XGBoost vs. Random Forest; desempeño por tipo de equipo; sensibilidad del horizonte temporal.

**Bloque 4 — Validación del modelo**
Validación cruzada estratificada de 5 folds (estabilidad frente al muestreo); experimento de estrategias de balanceo: ponderación por costo frente a SMOTENC total y parcial.

**Bloque 5 — Capa descriptiva**
Frecuencia de fallas por sistema; normalización y Pareto de componentes; heatmap de recurrencia temprana por tipo × sistema; serie temporal de fallas por tipo.

**Bloque 6 — Construcción de protocolos**
Matriz histórica de priorización por tipo × sistema, con recurrencia, frecuencia, componentes frecuentes y reemplazados, horas de parada (con cobertura) y nivel de evidencia. Orienta qué inspeccionar; no es el checklist operativo. Descarga consolidada de entregables.

## Equipos y sistemas cubiertos

**Tipos de equipo analizados**

| Nombre corto | Nombre completo                              |
|--------------|----------------------------------------------|
| RX Fijo      | Unidad Radiográfica Digital                  |
| RX Móvil     | Unidad Radiográfica Móvil                    |
| Mamografía   | Unidad Radiográfica Mamográfica              |
| Arco C       | Unidad Radiográfica/Fluoroscópica Móvil      |
| TAC          | Sis Exploración Tomografía Computarizada     |
| Fluoro       | Sistema radiográfico/fluoroscópico           |
| Angiografía  | Sist Radiográf/Fluorosc Para Angiografía     |

**Sistemas consolidados (variable `sistema_general`, 11 clases)**

Mecánico y de posicionamiento · Generación y detección de rayos X · Eléctrico del equipo · Control e interfaz de usuario · Procesamiento y almacenamiento · Comunicaciones · Seguridad del paciente · Accesorios · Otro · Falla desconocida · Usuario.

"Falla desconocida" (categoría residual no inspeccionable) y "Usuario" (remitida a una actividad transversal de capacitación) se excluyen de la matriz de priorización.

## Features del modelo

**Modelo A — Periodo de fallo (11 features: 10 numéricas + tipo de equipo)**

| Feature                    | Tipo       | Descripción                                          |
|----------------------------|------------|------------------------------------------------------|
| tbf_dias                   | Numérica   | Intervalo desde el fallo anterior (TBF actual)       |
| fallos_30_dias             | Numérica   | Fallos del equipo en los 30 días previos             |
| fallos_90_dias             | Numérica   | Fallos del equipo en los 90 días previos             |
| tbf_promedio_equipo        | Numérica   | TBF promedio histórico del equipo (fallos previos)   |
| tendencia_tbf              | Numérica   | Desviación del TBF previo respecto al histórico      |
| criticidad_temporal        | Numérica   | Índice de criticidad temporal, past-only (1–5)       |
| edad_dias                  | Numérica   | Días instalado al momento del fallo                  |
| horas_parada_acumuladas    | Numérica   | Horas fuera de servicio de fallos anteriores         |
| tasa_fallo_equipo          | Numérica   | Ritmo de fallo del equipo (fallos por año)           |
| tbf_anterior               | Numérica   | TBF del intervalo previo del equipo                  |
| tipo_equipo                | Categórica | Tipo de equipo médico                                |

`criticidad_temporal` (ventana expansiva anual, past-only) reemplaza a la criticidad global dentro del modelo para evitar fuga temporal entre años; la criticidad global se conserva solo como variable descriptiva y de priorización. Las tres features de mayor importancia (fallos en 90 días, TBF promedio y tasa de fallo) concentran cerca del 44% de la importancia total, lo que confirma que la actividad reciente de fallos es la señal más predictiva.

## Análisis exploratorio destacado

- Distribución del TBF fuertemente asimétrica a la derecha (asimetría ≈ 3.96 en la población objetivo) y de alta varianza (0 a 631 días), lo que justifica clasificación binaria sobre regresión.
- El 64% de los intervalos entre fallas registró una nueva falla dentro de los 30 días siguientes, lo que fundamenta el umbral del target y la clase "Requiere acción" (balance 64% / 36%).
- La curva de supervivencia empírica (1−ECDF sobre los intervalos de TBF) muestra que a los 30 días una mayoría de los intervalos ya presentó la falla siguiente.
- Coeficiente de silueta ≈ 0.011 sobre la proyección UMAP 2D: las clases no forman grupos separables por distancia. No es una prueba formal de no separabilidad, pero motiva evaluar modelos no lineales.

## Experimentos

| Experimento                         | Resultado                                              |
|-------------------------------------|-------------------------------------------------------|
| Comparación de 4 modelos (test)     | Empate técnico RF (0.658) vs. XGBoost (0.651)         |
| Validación temporal (walk-forward)  | Equivalencia estadística; XGBoost más estable         |
| Sensibilidad del horizonte          | 15/30/45/60/90 días: 30 días competitivo y estable    |
| Balanceo: SMOTENC total             | Aumenta la brecha train–test (mayor sobreajuste)      |
| Balanceo: SMOTENC parcial (60%)     | Sin mejora relevante de desempeño                     |

**Hallazgo clave.** Las tres estrategias de balanceo alcanzan un F1-macro equivalente; la diferencia decisiva es la brecha train–test, que el sobremuestreo sintético incrementa al memorizar puntos artificiales. Se adopta `compute_sample_weight("balanced")`, que equilibra las clases sin generar datos artificiales y preserva la distribución real de fallas. A esto se suma un criterio de dominio: interpolar variables entre órdenes de equipos distintos genera registros que no corresponden a eventos reales.

## Resultados visuales

- Análisis de TBF y fundamento del umbral — `assets/analisis_tbf.png`
- Distribución del target por tipo de equipo — `assets/distribucion_target.png`
- Separabilidad UMAP del target — `assets/separabilidad_umap.png`
- Comparativa de los 4 modelos (global y clase crítica) — `assets/comparativa_4_modelos_a.png`, `assets/comparativa_4_modelos_clase_a.png`
- Validación temporal comparativa — `assets/validacion_temporal_comparativa.png`
- Desempeño por tipo de equipo — `assets/desempeno_por_tipo.png`
- Sensibilidad del horizonte temporal — `assets/sensibilidad_horizonte.png`

## Matriz de priorización y checklist

El producto del Bloque 6 es la **matriz histórica de priorización** (`matriz_priorizacion.csv` / `.html`). Cada fila corresponde a un sistema de un tipo de equipo y reúne recurrencia temprana (`tbf_siguiente ≤ 30 días`), frecuencia de fallas, componentes frecuentes y reemplazados, horas de parada promedio (solo sobre registros con dato real, con indicador de cobertura) y nivel de evidencia.

Criterio de inclusión en la matriz: recurrencia ≥ 40% y evidencia ≥ Media (n_fallas ≥ 10); un componente dominante (≥5 casos, ≥25%) actúa como señal de refuerzo, no como condición obligatoria.

La matriz **no es** el checklist operativo. El checklist se elabora en una etapa posterior a partir de la matriz, los manuales del fabricante, los criterios de aceptación y el criterio de ingeniería clínica.

**Alcance del componente predictivo por tipo de equipo.** El desempeño no es homogéneo: en equipos con mayor volumen (RX Móvil, RX Fijo, TAC) el modelo alcanza un desempeño aceptable; el Arco C se clasifica como no apto (recall de la clase crítica 0.000 pese a muestra suficiente, asociado a su baja recurrencia del 34%); tipos como Fluoro presentan desempeño limitado por muestra reducida (n=45), no por incapacidad del modelo. En los tipos sin desempeño aceptable la priorización se apoya en el análisis histórico de la matriz.

## Requisitos

```
pandas
numpy
scikit-learn
xgboost
catboost
umap-learn
plotly
kaleido
matplotlib
squarify
rapidfuzz
imbalanced-learn
```

Instalar en Colab:

```
pip install xgboost catboost umap-learn plotly kaleido squarify rapidfuzz imbalanced-learn
```

## Cómo ejecutar

1. Abrir el notebook en Google Colab.
2. Ejecutar las celdas en orden secuencial (Bloques 1 → 6).
3. En la primera celda, subir el archivo Excel de órdenes de mantenimiento cuando se solicite.
4. Las figuras y la matriz de priorización (PNG, CSV y HTML) se descargan de forma consolidada en la celda final.

El dataset del hospital no se incluye en este repositorio por restricciones de confidencialidad clínica.

## Limitaciones

- Dataset de un único hospital (FVL, 2020–2025); la generalización a otras instituciones requeriría reentrenamiento.
- Registro incompleto de horas de parada: solo el 10,4% de los eventos (125 de 1.199), con cobertura desigual entre tipos. Se usa como información complementaria, nunca como criterio de priorización.
- `criticidad_temporal` corrige la fuga temporal entre años, pero conserva una fuga intra-anual acotada por usar ventana anual en lugar de atómica; se declara como limitación.
- La validación temporal no aplica purga entre folds; el posible solape de etiquetas en el límite de año se declara como limitación.
- Censura por cierre del estudio: los últimos eventos de cada equipo, sin falla posterior observada, se excluyen del entrenamiento.
- Tipos con muestra reducida o baja recurrencia (Fluoro, Arco C) tienen desempeño predictivo limitado o no apto.

## Trabajo futuro

- Mejorar la captura de horas de parada en la práctica de registro de la institución.
- Reducir la fuga intra-anual mediante una ventana atómica o purga temporal explícita.
- Ampliar el dataset con datos de nuevos equipos incorporados al hospital.
- Validación prospectiva de los checklists con el equipo de ingeniería clínica de FVL.
- Extensión de la metodología a equipos de funcionamiento no ionizante (ultrasonido, resonancia magnética).


## Autoras

**Laura Camila Vargas Delgado** · Ingeniería Biomédica · Universidad Autónoma de Occidente  
**Valeria Mosquera Amador** · Ingeniería Biomédica · Universidad Autónoma de Occidente  
Semillero RIDGE — Ingeniería Clínica, Salud Digital y Educación  
Proyecto de pasantía organizacional en colaboración con Fundación Valle del Lili · Cali, Colombia

---

## Licencia

El código de este repositorio está bajo licencia [MIT](LICENSE).

Los datos clínicos utilizados para entrenar los modelos son propiedad del hospital colaborador y no se incluyen en este repositorio. Los modelos entrenados tampoco se distribuyen por contener patrones derivados de información confidencial.
