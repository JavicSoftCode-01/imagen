import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime
from matplotlib.patches import Circle

# --- CONFIG ---
IMAGE_PATH = "morado.jpg"                 # <-- Cambia si tu imagen tiene otro nombre
OUTPUT_DIR = "result"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"{os.path.splitext(os.path.basename(IMAGE_PATH))[0]}_result.png")
ARUCO_REAL_SIZE_CM = 5.0                  # tamaño real del ArUco en cm (5x5 por defecto)

def ensure_output_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped, M

def formato_longitud_from_cm(cm_val):
    return f"{cm_val:.2f} cm"

def formato_area(area_cm2):
    return f"{area_cm2:.2f} cm²"

def detectar_forma(contour, approx_eps_factor, circularity_threshold=0.72):
    """
    Detecta forma basándose en approxPolyDP + circularidad.
    Devuelve (shape_name, approx)
    """
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, approx_eps_factor * peri, True)
    v = len(approx)

    area = cv2.contourArea(contour)
    circularity = 0.0
    if peri > 0:
        circularity = 4 * math.pi * area / (peri * peri)

    # Priorizar la detección de círculo por circularidad
    if circularity >= circularity_threshold:
        return "Círculo", approx

    if v == 3:
        return "Triángulo", approx
    elif v == 4:
        rect = cv2.minAreaRect(contour)
        (_, (w, h), _) = rect
        if w == 0 or h == 0:
            return "Rectángulo", approx
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio <= 1.15:
            return "Cuadrado", approx
        else:
            return "Rectángulo", approx
    elif v > 8:
        return "Polígono", approx

    return "Polígono", approx

def detectar_referencia(image, config):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    aruco_params = cv2.aruco.DetectorParameters()
    corners_list, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    if ids is None or len(ids) == 0:
        sys.exit("❌ [ERROR] No se detectó el marcador ArUco en la imagen.")

    # Tomamos el primer marcador detectado como referencia
    c = corners_list[0].reshape((4,2)).astype(float)
    lado1 = np.linalg.norm(c[1] - c[0])
    lado2 = np.linalg.norm(c[2] - c[1])
    lado3 = np.linalg.norm(c[3] - c[2])
    lado4 = np.linalg.norm(c[0] - c[3])

    avg_pixels = (lado1 + lado2 + lado3 + lado4) / 4.0
    pixels_per_cm = avg_pixels / config['ARUCO_REAL_SIZE_CM']

    print(f"✅ [INFO] ArUco detectado. Escala: {pixels_per_cm:.2f} px/cm")
    print(f"    Lados ArUco (px): {lado1:.1f}, {lado2:.1f}, {lado3:.1f}, {lado4:.1f}")
    print(f"    Promedio: {avg_pixels:.1f}px = {config['ARUCO_REAL_SIZE_CM']}cm")

    return image.copy(), pixels_per_cm, 'aruco', corners_list

def detectar_objetos(work_img, pixels_per_cm, config, mode, aruco_corners=None):
    """Versión con debug visual del proceso y filtros mejorados"""
    h_img, w_img = work_img.shape[:2]
    gray_w = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)

    print(f"[DEBUG] Procesando imagen {w_img}x{h_img}")

    # Múltiples métodos de detección binaria para robustez
    methods = []

    # Método 1: Threshold adaptativo
    adaptive_th = cv2.adaptiveThreshold(gray_w, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 21, 10)
    methods.append(("Adaptive Threshold", adaptive_th))

    # Método 2: Otsu
    blur = cv2.GaussianBlur(gray_w, (config['BLUR_KERNEL'], config['BLUR_KERNEL']), 0)
    _, otsu_th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    methods.append(("Otsu", otsu_th))

    # Método 3: Canny (bordes)
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (config['MORPH_KERNEL'], config['MORPH_KERNEL']))
    canny_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    methods.append(("Canny + Close", canny_closed))

    # Crear máscara de exclusión del ArUco (para que no genere un contorno grande)
    exclude_mask = np.zeros_like(gray_w)
    if mode == 'aruco' and aruco_corners:
        for corner_set in aruco_corners:
            pts = corner_set.reshape(4, 2).astype(int)
            center = np.mean(pts, axis=0).astype(int)
            expanded_pts = []
            for pt in pts:
                direction = pt - center
                expanded_pt = center + direction * 1.2
                expanded_pts.append(expanded_pt.astype(int))
            expanded_pts = np.array(expanded_pts)
            cv2.fillPoly(exclude_mask, [expanded_pts], 255)

    # Probar cada método y seleccionar el mejor por cantidad de objetos válidos
    best_method = None
    max_objects = 0

    for method_name, binary_img in methods:
        # limpiar binaria
        cleaned = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

        # aplicar máscara de exclusión del ArUco
        if mode == 'aruco':
            cleaned = cv2.bitwise_and(cleaned, cv2.bitwise_not(exclude_mask))

        cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_objects = []
        for c in cnts:
            area_px = cv2.contourArea(c)

            # filtros de tamaño
            if area_px < config['MIN_CONTOUR_AREA_PX']:
                continue
            if area_px > (w_img * h_img * 0.1):  # no más del 10% de la imagen
                continue
            if area_px < config['ABS_MIN_AREA_PX']:
                continue

            # filtros de forma y proporción
            rect = cv2.minAreaRect(c)
            (_, (w_px, h_px), _) = rect
            if w_px == 0 or h_px == 0:
                continue
            aspect_ratio = max(w_px, h_px) / min(w_px, h_px)
            if aspect_ratio > config['MAX_ASPECT_RATIO']:
                continue

            # perímetro y circularidad
            peri = cv2.arcLength(c, True)
            if peri > 0:
                circularity = 4 * math.pi * area_px / (peri * peri)
                if circularity < config['MIN_CIRCULARITY']:
                    # si es muy irregular, descartamos
                    continue

            # solidez (convex hull)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area_px / hull_area
                if solidity < config['MIN_SOLIDITY']:
                    continue

            # filtro por area en cm2
            area_cm2 = area_px / (pixels_per_cm * pixels_per_cm)
            if area_cm2 < config['MIN_AREA_CM2'] or area_cm2 > config['MAX_AREA_CM2']:
                continue

            valid_objects.append(c)

        print(f"  {method_name}: {len(cnts)} contornos totales, {len(valid_objects)} válidos")

        if len(valid_objects) > max_objects:
            max_objects = len(valid_objects)
            best_method = (method_name, cleaned, valid_objects)

    if best_method is None or max_objects == 0:
        print("[DEBUG] No se encontraron objetos con ningún método")
        return []

    method_name, best_binary, valid_contours = best_method
    print(f"[DEBUG] Mejor método: {method_name} con {len(valid_contours)} objetos")

    results = []
    for i, c in enumerate(valid_contours):
        area_px = cv2.contourArea(c)
        perim_px = cv2.arcLength(c, True)
        rect = cv2.minAreaRect(c)
        (center_px, (w_px, h_px), angle) = rect

        # Priorizar detección de forma con circularidad
        shape, approx = detectar_forma(c, config['APPROX_EPS_FACTOR'], circularity_threshold=config['CIRCULARITY_THRESHOLD'])

        # --- CÁLCULOS PARA CÍRCULO (más precisos) ---
        if shape == "Círculo":
            (x_c, y_c), radius_px = cv2.minEnclosingCircle(c)
            diameter_px = 2.0 * radius_px

            radius_cm = radius_px / pixels_per_cm
            diameter_cm = diameter_px / pixels_per_cm
            area_cm2 = math.pi * (radius_cm ** 2)
            perim_cm = 2 * math.pi * radius_cm

            cx, cy = int(round(x_c)), int(round(y_c))
            w_cm = h_cm = diameter_cm

            results.append({
                "shape": shape,
                "w_cm": w_cm,
                "h_cm": h_cm,
                "perim_cm": perim_cm,
                "area_cm2": area_cm2,
                "center": (cx, cy),
                "contour": c,
                "angle": angle,
                "radius_px": radius_px,
                "radius_cm": radius_cm
            })
            print(f"  Objeto {i+1}: {shape}, diámetro {diameter_cm:.2f} cm, Área: {area_cm2:.2f} cm², circularidad ~{4*math.pi*area_px/(perim_px*perim_px):.2f}")
            continue

        # --- CASO GENERAL (polígono/rectángulo) ---
        if h_px > w_px:
            w_px, h_px = h_px, w_px

        w_cm = w_px / pixels_per_cm
        h_cm = h_px / pixels_per_cm
        area_cm2 = area_px / (pixels_per_cm * pixels_per_cm)
        perim_cm = perim_px / pixels_per_cm

        cx, cy = int(round(center_px[0])), int(round(center_px[1]))

        results.append({
            "shape": shape,
            "w_cm": w_cm,
            "h_cm": h_cm,
            "perim_cm": perim_cm,
            "area_cm2": area_cm2,
            "center": (cx, cy),
            "contour": c,
            "angle": angle
        })

        print(f"  Objeto {i+1}: {shape}, {w_cm:.2f} x {h_cm:.2f} cm, Área: {area_cm2:.2f} cm²")

    print(f"[DEBUG] Resultado: {len(results)} objetos procesados")
    return results

def create_detailed_info_text(results, mode, pixels_per_cm):
    """
    Crea el texto de detalle que va abajo de la figura.
    Para CÍRCULO muestra: Diámetro, Perímetro, Área, Radio y Centro (añadido Radio)
    Para otros: Ancho, Alto, Perímetro, Área y Centro
    """
    lines = [
        "=" * 80,
        f"REPORTE DE MEDICIONES - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
        f"Método de calibración: ArUco {ARUCO_REAL_SIZE_CM}x{ARUCO_REAL_SIZE_CM} cm",
        f"Escala de conversión: {pixels_per_cm:.2f} píxeles/cm",
        f"Total de objetos detectados: {len(results)}",
        ""
    ]

    for i, obj in enumerate(results, 1):
        if obj['shape'] == "Círculo":
            # Para círculos: Diámetro, Perímetro, Área, Radio y Centro
            diameter = obj['w_cm']  # guardamos diámetro en w_cm para círculos
            lines.extend([
                f"OBJETO {i}: {obj['shape'].upper()}",
                f"  • Diámetro:   {formato_longitud_from_cm(diameter)}",
                f"  • Perímetro:  {formato_longitud_from_cm(obj['perim_cm'])}",
                f"  • Área:       {formato_area(obj['area_cm2'])}",
                f"  • Radio:      {formato_longitud_from_cm(obj.get('radius_cm', 0.0))}",
                f"  • Centro:     ({obj['center'][0]}, {obj['center'][1]}) px",
                ""
            ])
        else:
            # Para poligonos/rectángulos: Ancho, Alto, Perímetro, Área, Centro
            lines.extend([
                f"OBJETO {i}: {obj['shape'].upper()}",
                f"  • Ancho:      {formato_longitud_from_cm(obj['w_cm'])}",
                f"  • Alto:       {formato_longitud_from_cm(obj['h_cm'])}",
                f"  • Perímetro:  {formato_longitud_from_cm(obj['perim_cm'])}",
                f"  • Área:       {formato_area(obj['area_cm2'])}",
                f"  • Centro:     ({obj['center'][0]}, {obj['center'][1]}) px",
                ""
            ])

    return "\n".join(lines)

def create_measurement_plot(work_img, results, pixels_per_cm, mode, config, aruco_corners=None):
    """Crea el plot final con mediciones"""
    img_rgb = cv2.cvtColor(work_img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    fig_width = 10
    fig_height = fig_width * (h / w) + 3
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    ax.imshow(img_rgb)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    # Configurar ejes en cm (cada etiqueta cm_step cm)
    cm_step = 2
    x_ticks_px = np.arange(0, w + 1, pixels_per_cm * cm_step)
    y_ticks_px = np.arange(0, h + 1, pixels_per_cm * cm_step)
    x_labels = [f"{i*cm_step:.0f}" for i in range(len(x_ticks_px))]
    y_labels = [f"{i*cm_step:.0f}" for i in range(len(y_ticks_px))]

    ax.set_xticks(x_ticks_px)
    ax.set_xticklabels(x_labels)
    ax.set_yticks(y_ticks_px)
    ax.set_yticklabels(y_labels)

    ax.set_xlabel('Ancho (cm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Alto (cm)', fontsize=12, fontweight='bold')
    ax.set_title(f'Mediciones de Objetos - Referencia ArUco ({ARUCO_REAL_SIZE_CM}x{ARUCO_REAL_SIZE_CM} cm)',
                 fontsize=14, fontweight='bold', pad=18)

    # Dibujar ArUco
    if aruco_corners:
        for corner_set in aruco_corners:
            pts = corner_set.reshape(4, 2)
            ax.plot(np.append(pts[:, 0], pts[0, 0]),
                    np.append(pts[:, 1], pts[0, 1]),
                    color='lime', linewidth=3, alpha=0.9)
            center_aruco = np.mean(pts, axis=0)
            ax.text(center_aruco[0], center_aruco[1] - 20, 'ArUco\n5x5 cm',
                    fontsize=10, fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lime', alpha=0.8))

    # Colores para formas
    color_map = {
        'Rectángulo': 'red',
        'Cuadrado': 'blue',
        'Círculo': 'green',
        'Triángulo': 'orange',
        'Polígono': 'purple'
    }

    # Dibujar objetos con mediciones superpuestas
    for i, obj in enumerate(results, 1):
        contour_pts = obj['contour'].reshape(-1, 2)
        color = color_map.get(obj['shape'], 'gray')

        if obj['shape'] == 'Círculo' and 'radius_px' in obj:
            r_px = obj['radius_px']
            circ = Circle((obj['center'][0], obj['center'][1]), radius=r_px,
                          fill=False, linewidth=2.5, alpha=0.95, linestyle='-',
                          edgecolor=color)
            ax.add_patch(circ)
        else:
            ax.plot(np.append(contour_pts[:, 0], contour_pts[0, 0]),
                    np.append(contour_pts[:, 1], contour_pts[0, 1]),
                    color=color, linewidth=2, alpha=0.9)

        # Punto central
        ax.plot(obj['center'][0], obj['center'][1], 'o',
                color='white', markersize=8,
                markeredgecolor=color, markeredgewidth=2)

        # Número del objeto
        ax.text(obj['center'][0], obj['center'][1], str(i),
                fontsize=10, fontweight='bold', ha='center', va='center', color='black')

        # Texto de dimensiones:
        # Para CÍRCULO mostrar solo Perímetro y Radio (según tu pedido)
        if obj['shape'] == 'Círculo' and 'radius_cm' in obj:
            perim_cm = obj['perim_cm']
            radio_cm = obj['radius_cm']
            label_text = (f"Perímetro: {formato_longitud_from_cm(perim_cm)}\n"
                          f"Radio:     {formato_longitud_from_cm(radio_cm)}")
            ax.text(obj['center'][0], obj['center'][1] + 30, label_text,
                    fontsize=10, fontweight='bold', ha='center', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.85, edgecolor='white'),
                    color='white')
        else:
            # Para poligonos/rectángulos: Ancho, Alto, Área
            dimensions_text = f"Ancho: {formato_longitud_from_cm(obj['w_cm'])}\nAlto: {formato_longitud_from_cm(obj['h_cm'])}\nÁrea: {formato_area(obj['area_cm2'])}"
            ax.text(obj['center'][0], obj['center'][1] + 30, dimensions_text,
                    fontsize=10, fontweight='bold', ha='center', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.85, edgecolor='white'),
                    color='white')

    ax.grid(True, alpha=0.3)

    # Información detallada (abajo)
    info_text = create_detailed_info_text(results, mode, pixels_per_cm)
    plt.subplots_adjust(bottom=0.22)
    fig.text(0.05, 0.02, info_text, fontsize=9, fontfamily='monospace', va='bottom')

    return fig

def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        sys.exit(f"❌ [ERROR] No se pudo leer la imagen: {IMAGE_PATH}")

    h_img, w_img = img.shape[:2]

    # Configuración (ajustable)
    config = {
        'ARUCO_REAL_SIZE_CM': ARUCO_REAL_SIZE_CM,
        'MIN_CONTOUR_AREA_PX': 800,
        'ABS_MIN_AREA_PX': 2000,
        'BLUR_KERNEL': 5,
        'MORPH_KERNEL': 3,
        'APPROX_EPS_FACTOR': 0.02,
        'CIRCULARITY_THRESHOLD': 0.72,   # umbral para considerar círculo
        'MIN_CIRCULARITY': 0.05,         # filtro mínimo de circularidad para descartar formas muy irregulares
        'MIN_SOLIDITY': 0.5,
        'MAX_ASPECT_RATIO': 5.0,
        'MIN_AREA_CM2': 2.0,
        'MAX_AREA_CM2': 200.0
    }

    print(f"[INFO] Resolución: {w_img}x{h_img}")
    print(f"[INFO] Área mínima contorno (px): {config['MIN_CONTOUR_AREA_PX']}")

    # Detectar referencia ArUco y obtener escala
    work_img, pixels_per_cm, mode, aruco_corners = detectar_referencia(img, config)

    if pixels_per_cm is None or pixels_per_cm <= 0:
        sys.exit("❌ [ERROR] Escala inválida detectada (pixels_per_cm).")

    # Detectar objetos
    results = detectar_objetos(work_img, pixels_per_cm, config, mode, aruco_corners)

    print(f"✅ [INFO] {len(results)} objetos detectados y medidos.")

    # Guardar y mostrar resultado
    if results:
        ensure_output_dir(OUTPUT_PATH)
        fig = create_measurement_plot(work_img, results, pixels_per_cm, mode, config, aruco_corners)
        fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ [INFO] Imagen guardada en: {OUTPUT_PATH}")
        plt.show()
    else:
        print("⚠️ [WARN] No se detectaron objetos para medir.")

if __name__ == "__main__":
    main()
