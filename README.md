# Bank Marketing Campaign Analysis - Juan Mosquera Project 🏦

## Descripción del Proyecto

Este proyecto implementa un análisis completo de datos de marketing bancario utilizando la metodología CRISP-DM para predecir si un cliente suscribirá un depósito a término. El dataset proviene del UCI Machine Learning Repository y contiene información de campañas de telemarketing de un banco portugués entre mayo 2008 y noviembre 2010.

### Objetivo Principal
Desarrollar un modelo predictivo que determine la probabilidad de que un cliente suscriba un depósito a término basado en sus características demográficas, información de contacto y contexto económico.

## Diccionario de Datos Original 📊

El dataset original contiene **41,188 registros** con **20 variables** de entrada más 1 variable objetivo:

### Variables del Cliente
| Variable | Tipo | Descripción |
|----------|------|-------------|
| `age` | Numérica | Edad del cliente |
| `job` | Categórica | Tipo de trabajo (12 categorías: admin, blue-collar, entrepreneur, housemaid, management, retired, self-employed, services, student, technician, unemployed, unknown) |
| `marital` | Categórica | Estado civil (divorced, married, single, unknown) |
| `education` | Categórica | Nivel educativo (basic.4y, basic.6y, basic.9y, high.school, illiterate, professional.course, university.degree, unknown) |
| `default` | Categórica | ¿Tiene crédito en default? (yes, no, unknown) |
| `housing` | Categórica | ¿Tiene préstamo de vivienda? (yes, no, unknown) |
| `loan` | Categórica | ¿Tiene préstamo personal? (yes, no, unknown) |

### Variables de Contacto (Campaña Actual)
| Variable | Tipo | Descripción |
|----------|------|-------------|
| `contact` | Categórica | Tipo de comunicación (cellular, telephone) |
| `month` | Categórica | Mes del último contacto |
| `day_of_week` | Categórica | Día de la semana del último contacto |
| `duration` | Numérica | Duración del último contacto en segundos ⚠️ |

### Variables de Campaña
| Variable | Tipo | Descripción |
|----------|------|-------------|
| `campaign` | Numérica | Número de contactos realizados en esta campaña |
| `pdays` | Numérica | Días desde el último contacto de una campaña anterior (999 = no contactado previamente) |
| `previous` | Numérica | Número de contactos realizados antes de esta campaña |
| `poutcome` | Categórica | Resultado de la campaña anterior (failure, nonexistent, success) |

### Variables de Contexto Socioeconómico
| Variable | Tipo | Descripción |
|----------|------|-------------|
| `emp.var.rate` | Numérica | Tasa de variación del empleo (indicador trimestral) |
| `cons.price.idx` | Numérica | Índice de precios al consumidor (indicador mensual) |
| `cons.conf.idx` | Numérica | Índice de confianza del consumidor (indicador mensual) |
| `euribor3m` | Numérica | Tasa Euribor a 3 meses (indicador diario) |
| `nr.employed` | Numérica | Número de empleados (indicador trimestral) |

### Variable Objetivo
| Variable | Tipo | Descripción |
|----------|------|-------------|
| `y` | Binaria | ¿El cliente suscribió un depósito a término? (yes, no) |

## Decisiones Técnicas Implementadas en tools.py ⚙️

### 1. Eliminación de Variables Problemáticas
- **Variable `duration` eliminada**: Esta variable afecta altamente el target (si duration=0 entonces y="no"), pero no está disponible antes de realizar la llamada, por lo que no es útil para un modelo predictivo realista.

### 2. Tratamiento de Valores Faltantes
- **Estrategia adoptada**: Mantener "unknown" como una categoría válida en lugar de imputar o eliminar registros
- **Justificación**: Los valores "unknown" pueden contener información valiosa y representar un patrón de comportamiento específico

### 3. Feature Engineering Implementado
Se crearon **4 nuevas variables** basadas en el análisis exploratorio:

#### 3.1 Grupos de Edad (`age_group`)
```python
bins=[0, 30, 40, 50, 60, 100]
labels=['young', 'middle_young', 'middle_old', 'senior', 'elderly']
```

#### 3.2 Intensidad de Campaña (`campaign_intensity`)
```python
'low' if campaign == 1
'medium' if campaign <= 3
'high' if campaign > 3
```

#### 3.3 Historial de Contacto (`contact_history`)
```python
'first_contact' if previous == 0
'previous_success' if poutcome == 'success'
'previous_failure' if poutcome == 'failure'
'previous_unknown' otherwise
```

#### 3.4 Contexto Económico (`economic_context`)
```python
pd.cut(nr.employed, bins=3, labels=['low_employment', 'medium_employment', 'high_employment'])
```

### 4. Manejo de Multicolinealidad
- **Variables eliminadas**: `cons.price.idx`, `euribor3m`, `emp.var.rate`, `cons.conf.idx`
- **Variable retenida**: `nr.employed` como representante del contexto socioeconómico
- **Justificación**: Estas variables presentaban correlaciones > 0.9, generando redundancia en el modelo

### 5. Estrategia de Codificación
- **Variables categóricas**: Mantenidas para posterior One-Hot Encoding
- **Variables numéricas**: Preparadas para escalamiento estándar
- **Variable objetivo**: Codificada como binaria (0/1)

## Nuevo Diccionario de Datos - Variables Finales del Modelo 🎯

Después de la preparación de datos, el modelo utiliza **18 variables finales**:

### Variables Categóricas (13)
| Variable | Tipo | Descripción | Origen |
|----------|------|-------------|---------|
| `job` | Categórica | Tipo de trabajo | Original |
| `marital` | Categórica | Estado civil | Original |
| `education` | Categórica | Nivel educativo | Original |
| `default` | Categórica | Crédito en default | Original |
| `housing` | Categórica | Préstamo de vivienda | Original |
| `loan` | Categórica | Préstamo personal | Original |
| `contact` | Categórica | Tipo de comunicación | Original |
| `month` | Categórica | Mes del contacto | Original |
| `day_of_week` | Categórica | Día de la semana | Original |
| `poutcome` | Categórica | Resultado campaña anterior | Original |
| `age_group` | Categórica | Grupos etarios | **Engineered** |
| `campaign_intensity` | Categórica | Intensidad de campaña | **Engineered** |
| `contact_history` | Categórica | Historial de contacto | **Engineered** |
| `economic_context` | Categórica | Contexto económico | **Engineered** |

### Variables Numéricas (5)
| Variable | Tipo | Descripción | Origen |
|----------|------|-------------|---------|
| `age` | Numérica | Edad del cliente | Original |
| `campaign` | Numérica | Número de contactos | Original |
| `pdays` | Numérica | Días desde último contacto | Original |
| `previous` | Numérica | Contactos anteriores | Original |
| `nr.employed` | Numérica | Número de empleados | Original |

### Transformaciones Aplicadas
- **Variables eliminadas**: 6 (duration + 4 económicas correlacionadas + 1 redundante)
- **Variables engineered**: 4
- **Variables originales mantenidas**: 14
- **Forma final del dataset**: (41,188, 18)

## Arquitectura del Pipeline de Procesamiento 🔄

```python
# 1. Eliminación de variables problemáticas
df_prep = df.drop('duration', axis=1)

# 2. Feature Engineering
df_prep['age_group'] = pd.cut(...)
df_prep['campaign_intensity'] = df_prep['campaign'].apply(...)
df_prep['contact_history'] = df_prep.apply(create_contact_history, axis=1)
df_prep['economic_context'] = pd.cut(df_prep['nr.employed'], ...)

# 3. Eliminación de variables correlacionadas
economic_vars_to_remove = ['cons.price.idx', 'euribor3m', 'emp.var.rate', 'cons.conf.idx']
df_prep = df_prep.drop(economic_vars_to_remove, axis=1)

# 4. Selección final
X = df_prep[feature_vars]
y = df_prep['y'].map({'no': 0, 'yes': 1})
```

## Conclusiones del Proyecto 📈

### 1. Calidad de los Datos
- **Dataset robusto**: 41,188 registros con información completa y bien estructurada
- **Desbalance de clases**: La clase positiva (suscripción) representa aproximadamente el 11.27% de los datos
- **Valores faltantes**: Manejados estratégicamente manteniendo "unknown" como categoría informativa

### 2. Hallazgos del Análisis Exploratorio
- **Patrones demográficos**: Clientes mayores, jubilados y estudiantes muestran mayor propensión a suscribir
- **Influencia temporal**: Los meses de marzo, septiembre, octubre y diciembre presentan mejores tasas de conversión
- **Impacto del historial**: Clientes con campañas anteriores exitosas tienen significativamente mayor probabilidad de suscripción
- **Contexto económico**: Las variables socioeconómicas están altamente correlacionadas, sugiriendo que el contexto macroeconómico actúa como un factor unificado

### 3. Decisiones de Ingeniería de Features
- **Feature Engineering efectivo**: La creación de variables derivadas captura patrones no lineales importantes
- **Reducción de dimensionalidad**: La eliminación de variables correlacionadas mejora la interpretabilidad sin pérdida significativa de información
- **Encoding strategy**: La preservación de categorías "unknown" mantiene información valiosa sobre la incertidumbre

### 4. Preparación para Modelado
- **Dataset balanceado en features**: 13 variables categóricas y 5 numéricas proporcionan un balance adecuado
- **Escalabilidad**: El pipeline diseñado es eficiente para el volumen de datos y puede adaptarse a nuevos datos
- **Interpretabilidad**: Las transformaciones mantienen la interpretabilidad del modelo, crucial para decisiones de negocio

### 5. Recomendaciones para Implementación
- **Estrategia de validación**: Implementar validación temporal dado que los datos están ordenados por fecha
- **Monitoreo de drift**: Establecer monitoreo de las variables económicas que pueden cambiar significativamente
- **Segmentación**: Considerar modelos específicos por grupos demográficos dados los patrones diferenciados encontrados

### 6. Limitaciones Identificadas
- **Sesgo temporal**: Los datos cubren un período específico (2008-2010) que incluye la crisis financiera
- **Generalización**: Los patrones encontrados son específicos del contexto portugués y bancario
- **Variables no disponibles**: Información sobre ingresos del cliente o historial crediticio completo podría mejorar las predicciones



---

## Información Técnica

**Desarrollado por**: Juan Mosquera  
**Dataset**: UCI Bank Marketing Dataset  
**Metodología**: CRISP-DM  
**Herramientas**: Python, Pandas, Scikit-learn, Matplotlib, Seaborn  
**Fecha**: Septiembre 2024  

---

*Este README documenta las decisiones técnicas implementadas en `tools.py` y proporciona una guía completa para entender la preparación y transformación de datos realizadas en el proyecto.*