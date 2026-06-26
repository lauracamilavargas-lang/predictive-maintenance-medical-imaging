# Sistema de Mantenimiento Predictivo — Equipos de Imagenología Médica

> Pipeline de Machine Learning para predecir fallos en equipos de imagenología médica mediante una arquitectura **Classifier Chain** de dos modelos XGBoost encadenados.

**Desarrollado en:** Universidad Autónoma de Occidente · Ingeniería Biomédica  
**Período del dataset:** 2020 – 2025  
**Entorno de ejecución:** Google Colab

---

## Descripción general

Este sistema analiza **857 órdenes de mantenimiento correctivo** de cuatro tipos de equipos de imagenología (Arco C, RX Fijo, RX Móvil y Mamografía) para predecir, ante un nuevo evento de fallo:

1. **¿Cuándo volverá a fallar el equipo?** → clasificado en tres niveles de urgencia operacional.
2. **¿Qué sistema del equipo fallará?** → identificado entre 10 categorías técnicas.

La predicción del Modelo A (periodo de fallo) se incorpora como *feature* adicional al Modelo B (sistema afectado), formando una cadena de clasificadores que captura la dependencia entre urgencia y sistema involucrado.

---

## Resultados

| Modelo | Target | Clases | F1-macro baseline | F1-macro XGBoost | Mejora |
|--------|--------|--------|:-----------------:|:----------------:|:------:|
| **Modelo A** | Periodo de fallo | 3 | 0.212 | **0.834** | +0.622 |
| **Modelo B** | Sistema afectado | 10 | 0.031 | **0.828** | +0.797 |

**Exact Match conjunto** (ambos modelos aciertan en el mismo registro): **0.731**

---

## Arquitectura del pipeline

```
Excel del hospital
        │
        ▼
┌─────────────────────────────────┐
│  Bloque 1 — Pre-procesamiento   │
│  · Limpieza y estandarización   │
│  · Cálculo de TBF               │
│  · Índice de criticidad (1-5)   │
│  · Features temporales          │
│  · df_modelo: 1,003 registros   │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Bloque 2 — EDA                 │
│  · Distribución de targets      │
│  · Correlaciones de Pearson     │
│  · PCA 2D/3D (Modelo A y B)     │
│  · UMAP 2D/3D (Modelo A y B)    │
│  · Curva de supervivencia       │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Bloque 3 — Modelado            │
│  · DummyClassifier (baseline)   │
│  · Modelo A: XGBoost 3 clases   │
│  · Modelo B: XGBoost 10 clases  │
│  · Experimentos adicionales     │
│    (Classifier Chain, CatBoost) │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Bloque 4 — Protocolos          │
│  · Tabla maestra tipo×urgencia  │
│  · Semáforo de confianza        │
│  · Check lists por equipo       │
└─────────────────────────────────┘
```

---

## 📊 Equipos y sistemas cubiertos

**Tipos de equipo analizados:**

| Nombre corto | Nombre completo |
|--------------|-----------------|
| RX Fijo | Unidad Radiográfica Digital |
| RX Móvil | Unidad Radiográfica Móvil |
| Mamografía | Unidad Radiográfica Mamográfica |
| Arco C | Unidad Radiográfica/Fluoroscópica Móvil |
| TAC | Sis Exploración Tomografía Computarizada |
| Fluoroscopía | Sistema radiográfico/fluoroscópico |
| Angiografía | Sist Radiográf/Fluorosc Para Angiografía |

**Sistemas afectados (clases Modelo B):**

- Mecánico y de posicionamiento
- Generación y detección de rayos X
- Eléctrico del equipo
- Control e interfaz de usuario
- Procesamiento y almacenamiento
- Comunicaciones
- Falla desconocida
- Seguridad del paciente
- Usuario
- Otro

---

## 🛠️ Features del modelo

### Modelo A — Periodo de fallo (10 features)

| Feature | Tipo | Descripción |
|---------|------|-------------|
| `fallos_30_dias` | Numérica | Fallos del equipo en los últimos 30 días |
| `fallos_90_dias` | Numérica | Fallos del equipo en los últimos 90 días |
| `tbf_promedio_equipo` | Numérica | TBF promedio histórico hasta ese evento |
| `tendencia_tbf` | Numérica | Desviación del TBF actual vs. histórico |
| `tasa_fallo_equipo` | Numérica | Fallos acumulados / edad en días |
| `dias_desde_ultimo_fallo` | Numérica | Días desde el fallo anterior |
| `criticidad` | Numérica | Índice compuesto de gravedad (1–5) |
| `edad_dias` | Numérica | Días instalado al momento del fallo |
| `horas_parada_acumuladas` | Numérica | Total horas fuera de servicio acumuladas |
| `tipo_equipo` | Categórica | Tipo de equipo médico |

**Variables más importantes:** `fallos_90_dias` (0.427) + `fallos_30_dias` (0.275) = 70% de la importancia total.

### Modelo B — Sistema afectado (10 features)

| Feature | Tipo | Descripción |
|---------|------|-------------|
| `componente_agrupado` | Categórica | Sistema al que pertenece el componente que falló |
| `tipo_equipo` | Categórica | Tipo de equipo médico |
| `tipo_solucion` | Categórica | Tipo de solución aplicada |
| `hubo_cambio_componente` | Binaria | Si se reemplazó algún componente |
| `criticidad` | Numérica | Índice de gravedad (1–5) |
| `fallos_30_dias` | Numérica | Fallos en los últimos 30 días |
| `edad_dias` | Numérica | Días instalado al momento del fallo |
| `tbf_promedio_equipo` | Numérica | TBF promedio histórico |
| `tendencia_tbf` | Numérica | Deterioro o mejora reciente |
| `horas_parada_acumuladas` | Numérica | Total horas fuera de servicio |

**Variable más importante:** `componente_agrupado` (0.287) — el componente físico es la señal más directa del sistema afectado.

---

## 📈 Análisis exploratorio destacado

- **Asimetría TBF = 4.31** → justifica clasificación sobre regresión
- **P(fallo ≤ 15 días) = 42.9%** → caída casi vertical en curva de supervivencia
- **P(fallo ≤ 90 días) = 89.7%** → el 90% de los equipos falla antes de 3 meses
- **Silhouette Score negativo** (PCA y UMAP) → no hay separabilidad lineal entre clases → justifica XGBoost no lineal
- **Correlación Modelo A:** `fallos_30_dias` (−0.567) y `fallos_90_dias` (−0.501) son las más predictivas
- **Correlación Modelo B:** ninguna feature supera 0.16 lineal → patrones capturados por relaciones no lineales

---

## Experimentos adicionales

| Experimento | F1-macro Modelo B | Resultado |
|-------------|:-----------------:|-----------|
| XGBoost independiente | **0.828** | ✅ Mejor resultado |
| Classifier Chain A→B | 0.767 | ↓ No mejora — targets independientes |
| Classifier Chain B→A | 0.754 | ↓ No mejora |
| CatBoost | 0.806 | ↓ XGBoost más robusto globalmente |

> **Hallazgo clave:** Los targets `periodo_fallo` y `sistema_general` son prácticamente independientes (correlación ~0). Predecir cuándo fallará un equipo no ayuda a predecir qué sistema fallará, lo que convierte la Classifier Chain en un experimento válido aunque sin ganancia en este dataset.

---

## 📂 Archivos generados

| Archivo | Descripción |
|---------|-------------|
| `tabla_maestra_protocolos.csv` | Base empírica: tipo × urgencia × sistema con métricas históricas |
| `tabla_maestra_con_semaforo.csv` | Tabla anterior + semáforo de confianza por combinación |
| `tabla_reclasificada.csv` | Cada sistema en su única categoría de riesgo principal |
| `tabla_reclasificada_f1.csv` | Tabla anterior + semáforo basado en F1 del Modelo B |
| `checklist_base.csv` | Check list completa por tipo de equipo con acciones y tiempos |

---

## Semáforo de confianza

Los protocolos y check lists se asignan a una de tres categorías de confianza basadas en el F1 del Modelo B por sistema:

| Semáforo | Umbral F1 | Acción recomendada |
|----------|:---------:|--------------------|
| 🟢 **VERDE** | ≥ 0.85 | Protocolo activable automáticamente |
| 🟡 **AMARILLO** | 0.70 – 0.84 | Check list con validación del técnico |
| 🔴 **ROJO** | < 0.70 | Check list preventiva genérica + criterio experto |

---

## Requisitos

```python
pandas
numpy
scikit-learn
xgboost
catboost
umap-learn
plotly
matplotlib
```

Instalar en Colab:

```bash
pip install xgboost catboost umap-learn plotly
```

---

## Cómo ejecutar

1. Abrir `Versión_final_modelo_de_mtto_predictivo.ipynb` en Google Colab.
2. Ejecutar las celdas en orden secuencial (Bloques 1 → 2 → 3 → 4).
3. Al llegar a la **Celda 1**, subir el archivo Excel de órdenes de mantenimiento del hospital cuando se solicite.
4. Al llegar a la **Celda 31**, subir el mismo archivo nuevamente para la generación de check lists.
5. Los archivos CSV de resultados se guardan automáticamente en `/content/`.

> El dataset del hospital no se incluye en este repositorio por restricciones de confidencialidad clínica.

---

## Limitaciones

- Dataset pequeño (1,003 registros, 10 clases en Modelo B)
- Clases minoritarias con pocos registros: `usuario` (30) y `seguridad del paciente` (37)
- Los modelos están entrenados sobre datos de un único hospital — la generalización a otras instituciones requiere reentrenamiento

---

## Trabajo futuro

- [ ] Ampliar dataset a ~2,000 registros con datos reales
- [ ] Aplicar SMOTE para clases minoritarias
- [ ] Ajuste de hiperparámetros con cross-validation anidada
- [ ] Despliegue como API REST para integración con sistemas CMMS del hospital
- [ ] Extensión a TAC y equipos de angiografía con mayor volumen de datos

---

## 📄 Licencia

Proyecto académico desarrollado en la Universidad Autónoma de Occidente. Los datos clínicos utilizados son propiedad del hospital colaborador y no se distribuyen en este repositorio.
