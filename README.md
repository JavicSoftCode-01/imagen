# Medición de Formas en Imágenes

Este proyecto utiliza Python y OpenCV para detectar y medir formas geométricas en imágenes. La aplicación usa un marcador ArUco para establecer una escala de conversión (píxeles a centímetros), permitiendo medir con precisión las dimensiones de los objetos presentes en la imagen.

## Funcionalidades

- **Detección de Formas:** Identifica y clasifica automáticamente formas como círculos, triángulos, rectángulos, cuadrados y otros polígonos.
- **Calibración con ArUco:** Utiliza un marcador ArUco (por defecto de 5x5 cm) para rectificar la perspectiva y calcular la escala de la imagen.
- **Mediciones Detalladas:** Calcula dimensiones como perímetro, área, longitudes de lados, radio y diámetro, entre otros parámetros.
- **Reporte de Mediciones:** Genera un reporte detallado con la información calculada y guarda una imagen anotada con las mediciones.
- **Gráficos y Anotaciones:** Visualiza la imagen original con superposiciones que indican las mediciones y las formas detectadas.

## Requisitos

- Python 3.x
- OpenCV (opencv-python)
- NumPy
- Matplotlib

Las dependencias se pueden instalar ejecutando:

```bash
pip install -r requirements.txt
```

## Uso

1. Coloca la imagen a analizar en el mismo directorio y actualiza la variable `IMAGE_PATH` en el archivo `measure_shapes.py` si es necesario.
2. Asegúrate de tener el marcador ArUco en la imagen para la calibración.
3. Ejecuta el script:

```bash
python measure_shapes.py
```

El script procesará la imagen, detectará las formas, calculará las dimensiones y guardará la imagen resultante en la carpeta `result`.

## Configuración

El archivo `measure_shapes.py` contiene varios parámetros configurables, tales como:

- `ARUCO_REAL_SIZE_CM`: Tamaño real del marcador ArUco (por defecto 5 cm).
- Parámetros de filtrado y umbrales para la detección de contornos y formas.

Modifica estos parámetros según la calidad de imagen y los requerimientos de tu aplicación.

## Notas

- Se imprime información de debug en la consola para ayudar en la validación de la detección y las mediciones.
- Si la imagen no se puede leer o el marcador ArUco no es detectado, el script abortará y mostrará un mensaje de error.

---

Actualizado en Español basándose en la implementación de `measure_shapes.py`.