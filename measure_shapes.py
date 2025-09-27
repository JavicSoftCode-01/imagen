import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime
from matplotlib.patches import Circle
import string

# ----------------- CONFIG -----------------
IMAGE_PATH = "cuadrado.jpg"                 # <-- Cambia si tu imagen tiene otro nombre
OUTPUT_DIR = "result"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"{os.path.splitext(os.path.basename(IMAGE_PATH))[0]}_result.png")
ARUCO_REAL_SIZE_CM = 5.0                  # tamaño real del ArUco en cm (5x5 por defecto)
# ------------------------------------------

def ensure_output_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def formato_longitud_from_cm(cm_val):
    return f"{cm_val:.2f} cm"

def formato_area(area_cm2):
    return f"{area_cm2:.2f} cm²"

def _distance(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def _triangle_area_from_pts(p1, p2, p3):
    x1,y1 = p1; x2,y2 = p2; x3,y3 = p3
    return abs((x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2.0)

# ---------------- DETECCIÓN DE FORMA (MEJORADA) ----------------
def detectar_forma(contour, approx_eps_factor, circularity_threshold=0.8):
    """
    Lógica de decisión actualizada:
      - Primero calcular circularidad; si >= threshold -> Círculo (prioridad para formas circulares aunque approx de muchos vértices)
      - Luego approxPolyDP para v
      - si v==3 -> Triángulo
      - si v==4 -> Rectángulo/Cuadrado
      - si v>=5 -> Polígono
      - fallback -> Polígono
    Devuelve (shape_name, approx, v, circularity) (v y circularity para debug).
    """
    peri = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    circularity = 0.0
    if peri > 0:
        circularity = 4 * math.pi * area / (peri * peri)

    # Prioridad a círculo si circularidad alta
    if circularity >= circularity_threshold:
        return "Círculo", None, 0, circularity

    # Approx para otras formas
    approx = None
    eps_factors = [approx_eps_factor * 1.5, approx_eps_factor * 1.2, approx_eps_factor, approx_eps_factor*0.8, approx_eps_factor*0.6, 0.01]  # Añadir más altos para mejor aproximación
    for ef in eps_factors:
        a = cv2.approxPolyDP(contour, max(0.001, ef * peri), True)
        if a is not None and len(a) >= 3:
            approx = a
            break

    v = len(approx) if approx is not None else 0

    # Debug print
    print(f"[DEBUG-FORM] contorno: v={v}, circularity={circularity:.3f}, area_px={area:.1f}, peri={peri:.1f}")

    if v == 3:
        return "Triángulo", approx, v, circularity
    if v == 4:
        rect = cv2.minAreaRect(contour)
        (_, (w, h), _) = rect
        if w == 0 or h == 0:
            return "Rectángulo", approx, v, circularity
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio <= 1.12:
            return "Cuadrado", approx, v, circularity
        else:
            return "Rectángulo", approx, v, circularity
    if v >= 5:
        return "Polígono", approx, v, circularity
    return "Polígono", approx, v, circularity

# ---------------- Referencia ArUco con Rectificación de Perspectiva y Soporte Múltiple ----------------
def detectar_referencia(image, config):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    aruco_params = cv2.aruco.DetectorParameters()
    corners_list, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    if ids is None or len(ids) == 0:
        sys.exit("❌ [ERROR] No se detectó ningún marcador ArUco en la imagen.")
    
    # Usar el primer ArUco para homografía (asumiendo es el principal)
    c = corners_list[0].reshape((4,2)).astype(np.float32)
    
    # Calcular lados para debug del principal
    lado1 = np.linalg.norm(c[1] - c[0])
    lado2 = np.linalg.norm(c[2] - c[1])
    lado3 = np.linalg.norm(c[3] - c[2])
    lado4 = np.linalg.norm(c[0] - c[3])
    avg_pixels = (lado1 + lado2 + lado3 + lado4) / 4.0
    
    # Rectificación: Definir puntos ideales para ArUco cuadrado
    target_size_px = avg_pixels  # Mantener tamaño similar para preservar resolución
    ideal_corners = np.array([
        [0, 0],
        [target_size_px, 0],
        [target_size_px, target_size_px],
        [0, target_size_px]
    ], dtype=np.float32)
    
    # Ordenar corners detectados en orden consistente (top-left, top-right, bottom-right, bottom-left)
    sum_coords = np.sum(c, axis=1)
    sorted_idx = np.argsort(sum_coords)
    ordered_corners = np.array([c[sorted_idx[0]], c[sorted_idx[1]], c[sorted_idx[3]], c[sorted_idx[2]]], dtype=np.float32)
    
    # Computar homografía
    H, _ = cv2.findHomography(ordered_corners, ideal_corners)
    
    # Warp la imagen completa
    h_img, w_img = image.shape[:2]
    # Para evitar recorte, calcular bounding box de la imagen warpeada
    corners_img = np.array([[0,0], [w_img-1,0], [w_img-1,h_img-1], [0,h_img-1]], dtype=np.float32)
    warped_corners_img = cv2.perspectiveTransform(corners_img.reshape(-1,1,2), H).reshape(-1,2)
    min_x, min_y = np.min(warped_corners_img, axis=0)
    max_x, max_y = np.max(warped_corners_img, axis=0)
    warp_w = int(np.ceil(max_x - min_x))
    warp_h = int(np.ceil(max_y - min_y))
    translation_mat = np.array([[1,0,-min_x], [0,1,-min_y], [0,0,1]])
    H_adjusted = translation_mat @ H
    
    warped_img = cv2.warpPerspective(image, H_adjusted, (warp_w, warp_h), flags=cv2.INTER_LINEAR)
    
    # Warp todas las corners de TODOS los ArUcos detectados
    all_warped_aruco_corners = []
    all_warped_avg_pixels = []
    for corners in corners_list:
        c_orig = corners.reshape((4,2)).astype(np.float32)
        sum_coords_orig = np.sum(c_orig, axis=1)
        sorted_idx_orig = np.argsort(sum_coords_orig)
        ordered_corners_orig = np.array([c_orig[sorted_idx_orig[0]], c_orig[sorted_idx_orig[1]], c_orig[sorted_idx_orig[3]], c_orig[sorted_idx_orig[2]]], dtype=np.float32)
        warped_c = cv2.perspectiveTransform(ordered_corners_orig.reshape(-1,1,2), H_adjusted).reshape(-1,2)
        all_warped_aruco_corners.append(warped_c)
        
        # Calcular lados post-rect para cada uno
        wl1 = np.linalg.norm(warped_c[1] - warped_c[0])
        wl2 = np.linalg.norm(warped_c[2] - warped_c[1])
        wl3 = np.linalg.norm(warped_c[3] - warped_c[2])
        wl4 = np.linalg.norm(warped_c[0] - warped_c[3])
        w_avg = (wl1 + wl2 + wl3 + wl4) / 4.0
        all_warped_avg_pixels.append(w_avg)
    
    # Escala: Promedio de todos los ArUcos post-rect
    pixels_per_cm = np.mean(all_warped_avg_pixels) / config['ARUCO_REAL_SIZE_CM']
    
    print(f"✅ [INFO] {len(corners_list)} ArUco(s) detectado(s). Escala promedio: {pixels_per_cm:.2f} px/cm (post-rectificación)")
    print(f"    Lados ArUco principal originales (px): {lado1:.1f}, {lado2:.1f}, {lado3:.1f}, {lado4:.1f}")
    print(f"    Promedio principal original: {avg_pixels:.1f}px = {config['ARUCO_REAL_SIZE_CM']}cm")
    
    # Para debug, imprimir lados rectificados de todos
    for i, w_avg in enumerate(all_warped_avg_pixels):
        print(f"    ArUco {i+1} rectificado promedio: {w_avg:.1f} px")
    
    return warped_img, pixels_per_cm, 'aruco', all_warped_aruco_corners  # Retornar todos los warped corners

# ------------- Obtener vértices para polígonos -------------
def _get_polygon_vertices_from_contour(c, approx_eps_factor, max_vertices=30):
    peri = cv2.arcLength(c, True)
    eps_list = [approx_eps_factor * f for f in [2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2]]  # Más altos para reducir vértices
    for eps in eps_list:
        approx = cv2.approxPolyDP(c, max(0.001, eps) * peri, True)
        if approx is not None:
            pts = approx.reshape(-1,2).astype(float)
            if 3 <= len(pts) <= max_vertices:
                return pts
    hull = cv2.convexHull(c)
    hull_pts = hull.reshape(-1,2).astype(float)
    n = len(hull_pts)
    if 3 <= n <= max_vertices:
        return hull_pts
    for f in [0.05, 0.08, 0.1, 0.12]:  # Más altos
        approx = cv2.approxPolyDP(hull, f * peri, True)
        pts = approx.reshape(-1,2).astype(float)
        if 3 <= len(pts) <= max_vertices:
            return pts
    if n >= 3:
        step = max(1, n // max_vertices)
        sampled = hull_pts[::step][:max_vertices]
        if len(sampled) >= 3:
            return sampled
    cont_pts = c.reshape(-1,2).astype(float)
    if len(cont_pts) >= 3:
        return cont_pts[[0, len(cont_pts)//2, -1]]
    return np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0]], dtype=float)

def _order_vertices_clockwise(pts):
    cx = np.mean(pts[:,0]); cy = np.mean(pts[:,1])
    angles = np.arctan2(pts[:,1]-cy, pts[:,0]-cx)
    idx = np.argsort(angles)
    return pts[idx]

# -------------- Detección de objetos y cálculo --------------
def detectar_objetos(work_img, pixels_per_cm, config, mode, aruco_corners=None):
    h_img, w_img = work_img.shape[:2]
    gray_w = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
    print(f"[DEBUG] Procesando imagen {w_img}x{h_img}")

    # Métodos binarios (añadir bilateral filter para reducir ruido)
    gray_w = cv2.bilateralFilter(gray_w, 5, 75, 75)  # Reducir ruido preservando bordes

    methods = []
    adaptive_th = cv2.adaptiveThreshold(gray_w, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 5)
    methods.append(("Adaptive Threshold", adaptive_th))
    blur = cv2.GaussianBlur(gray_w, (config['BLUR_KERNEL'], config['BLUR_KERNEL']), 0)
    _, otsu_th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    methods.append(("Otsu", otsu_th))
    edges = cv2.Canny(blur, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config['MORPH_KERNEL'], config['MORPH_KERNEL']))
    canny_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    methods.append(("Canny + Close", canny_closed))

    # máscara excluir TODOS los ArUcos
    exclude_mask = np.zeros_like(gray_w)
    if mode == 'aruco' and aruco_corners:
        for corner_set in aruco_corners:
            pts = corner_set.astype(int)
            center = np.mean(pts, axis=0).astype(int)
            expanded_pts = pts + (pts - center) * 0.1
            cv2.fillPoly(exclude_mask, [expanded_pts.astype(int)], 255)

    # seleccionar mejor método
    best_method = None
    max_objects = 0
    for method_name, binary_img in methods:
        cleaned = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
        if mode == 'aruco':
            cleaned = cv2.bitwise_and(cleaned, cv2.bitwise_not(exclude_mask))
        cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        valid_objects = []
        for c in cnts:
            area_px = cv2.contourArea(c)
            if area_px < config['MIN_CONTOUR_AREA_PX']:
                continue
            if area_px > (w_img * h_img * 0.1):
                continue
            if area_px < config['ABS_MIN_AREA_PX']:
                continue
            rect = cv2.minAreaRect(c)
            (_, (w_px, h_px), _) = rect
            if w_px == 0 or h_px == 0:
                continue
            aspect_ratio = max(w_px, h_px) / min(w_px, h_px)
            if aspect_ratio > config['MAX_ASPECT_RATIO']:
                continue
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area_px / hull_area
                if solidity < config['MIN_SOLIDITY']:
                    continue
            area_cm2 = area_px / (pixels_per_cm ** 2)
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

        # detectar forma
        shape, approx, v, circularity = detectar_forma(c, config['APPROX_EPS_FACTOR'], circularity_threshold=config['CIRCULARITY_THRESHOLD'])

        # Forzar rectángulo si cerca de 4 vértices y alta solidez
        if shape == "Polígono" and 4 < v <= 8:
            solidity = area_px / cv2.contourArea(cv2.convexHull(c))
            aspect_ratio = max(w_px, h_px) / min(w_px, h_px)
            if solidity > 0.95 and aspect_ratio < 3.0:
                shape = "Rectángulo" if aspect_ratio > 1.12 else "Cuadrado"
                print(f"[DEBUG] Forzando {shape} para contorno con v={v}, solidity={solidity:.3f}")

        # CÍRCULO
        if shape == "Círculo":
            (x_c, y_c), radius_px = cv2.minEnclosingCircle(c)
            radius_cm = radius_px / pixels_per_cm
            diameter_cm = 2 * radius_cm
            area_cm2 = math.pi * (radius_cm ** 2)
            perim_cm = 2 * math.pi * radius_cm
            cx, cy = int(round(x_c)), int(round(y_c))
            results.append({
                "shape": shape,
                "w_cm": diameter_cm,
                "h_cm": diameter_cm,
                "perim_cm": perim_cm,
                "area_cm2": area_cm2,
                "center": (cx, cy),
                "contour": c,
                "angle": angle,
                "radius_px": radius_px,
                "radius_cm": radius_cm
            })
            print(f"  Objeto {i+1}: {shape}, radius_cm={radius_cm:.3f}, perim_cm={perim_cm:.3f}, area_cm2={area_cm2:.3f}")
            continue

        # TRIÁNGULO
        if shape == "Triángulo":
            verts = _get_polygon_vertices_from_contour(c, config['APPROX_EPS_FACTOR'], max_vertices=3)
            verts = _order_vertices_clockwise(verts)
            d01_px = _distance(verts[0], verts[1])
            d12_px = _distance(verts[1], verts[2])
            d20_px = _distance(verts[2], verts[0])
            sideA_cm = d01_px / pixels_per_cm
            sideB_cm = d12_px / pixels_per_cm
            sideC_cm = d20_px / pixels_per_cm
            sides_cm = [sideA_cm, sideB_cm, sideC_cm]
            sides_px = [d01_px, d12_px, d20_px]
            perim_cm = sum(sides_cm)
            area_cm2 = area_px / (pixels_per_cm ** 2)
            cx_int, cy_int = int(round(center_px[0])), int(round(center_px[1]))
            results.append({
                "shape": shape,
                "sides_cm": sides_cm,
                "sides_px": sides_px,
                "perim_cm": perim_cm,
                "area_cm2": area_cm2,
                "center": (cx_int, cy_int),
                "contour": c,
                "angle": angle,
                "vertices_px": verts.astype(int).tolist()
            })
            print(f"  Objeto {i+1}: {shape}, Lados (cm): {sideA_cm:.2f}, {sideB_cm:.2f}, {sideC_cm:.2f}  Área: {area_cm2:.2f} cm²")
            continue

        # RECTÁNGULO / CUADRADO
        if shape in ("Rectángulo", "Cuadrado"):
            (cx_r, cy_r), (w_px_r, h_px_r), ang_r = rect
            w_cm = w_px_r / pixels_per_cm
            h_cm = h_px_r / pixels_per_cm
            area_cm2 = area_px / (pixels_per_cm ** 2)
            perim_cm = 2 * (w_cm + h_cm)
            cx_i, cy_i = int(round(cx_r)), int(round(cy_r))
            results.append({
                "shape": shape,
                "w_cm": w_cm,
                "h_cm": h_cm,
                "perim_cm": perim_cm,
                "area_cm2": area_cm2,
                "center": (cx_i, cy_i),
                "contour": c,
                "angle": ang_r,
                "rect": rect  # Añadir rect para plotting
            })
            print(f"  Objeto {i+1}: {shape}, Ancho {w_cm:.2f} cm, Alto {h_cm:.2f} cm, Área: {area_cm2:.2f} cm²")
            continue

        # POLÍGONO (n lados)
        verts = _get_polygon_vertices_from_contour(c, config['APPROX_EPS_FACTOR'], max_vertices=config['MAX_VERTICES'])
        verts = _order_vertices_clockwise(verts)
        n = len(verts)
        sides_px = []
        sides_cm = []
        for idx in range(n):
            a = verts[idx]; b = verts[(idx+1) % n]
            d_px = _distance(a, b)
            sides_px.append(d_px)
            sides_cm.append(d_px / pixels_per_cm)
        perim_cm = sum(sides_cm)
        area_cm2 = area_px / (pixels_per_cm ** 2)
        cx_int, cy_int = int(round(center_px[0])), int(round(center_px[1]))
        results.append({
            "shape": "Polígono",
            "vertices_px": verts.astype(int).tolist(),
            "sides_px": sides_px,
            "sides_cm": sides_cm,
            "perim_cm": perim_cm,
            "area_cm2": area_cm2,
            "center": (cx_int, cy_int),
            "contour": c,
            "angle": angle
        })
        print(f"  Objeto {i+1}: Polígono con {n} lados. Perímetro: {perim_cm:.2f} cm, Área: {area_cm2:.2f} cm²")

    print(f"[DEBUG] Resultado: {len(results)} objetos procesados")
    return results

# ----------------- Texto de reporte -----------------
def create_detailed_info_text(results, mode, pixels_per_cm):
    lines = [
        "=" * 80,
        f"REPORTE DE MEDICIONES - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
        f"Método de calibración: ArUco {ARUCO_REAL_SIZE_CM}x{ARUCO_REAL_SIZE_CM} cm (con rectificación de perspectiva)",
        f"Escala de conversión: {pixels_per_cm:.2f} píxeles/cm",
        f"Total de objetos detectados: {len(results)}",
        ""
    ]
    for i, obj in enumerate(results, 1):
        if obj['shape'] == "Círculo":
            lines.extend([
                f"OBJETO {i}: {obj['shape'].upper()}",
                f"  • Radio:      {formato_longitud_from_cm(obj.get('radius_cm', obj.get('w_cm',0.0)/2))}",
                f"  • Diámetro:   {formato_longitud_from_cm(obj.get('w_cm', 0.0))}",
                f"  • Perímetro:  {formato_longitud_from_cm(obj['perim_cm'])}",
                f"  • Área:       {formato_area(obj['area_cm2'])}",
                f"  • Centro:     ({obj['center'][0]}, {obj['center'][1]}) px",
                ""
            ])
        elif obj['shape'] == "Triángulo":
            s = obj['sides_cm']
            lines.extend([
                f"OBJETO {i}: {obj['shape'].upper()}",
                f"  • Lado A:     {formato_longitud_from_cm(s[0])}",
                f"  • Lado B:     {formato_longitud_from_cm(s[1])}",
                f"  • Lado C:     {formato_longitud_from_cm(s[2])}",
                f"  • Perímetro:  {formato_longitud_from_cm(obj['perim_cm'])}",
                f"  • Área:       {formato_area(obj['area_cm2'])}",
                f"  • Centro:     ({obj['center'][0]}, {obj['center'][1]}) px",
                ""
            ])
        elif obj['shape'] in ("Rectángulo", "Cuadrado"):
            lines.extend([
                f"OBJETO {i}: {obj['shape'].upper()}",
                f"  • Ancho:      {formato_longitud_from_cm(obj['w_cm'])}",
                f"  • Alto:       {formato_longitud_from_cm(obj['h_cm'])}",
                f"  • Perímetro:  {formato_longitud_from_cm(obj['perim_cm'])}",
                f"  • Área:       {formato_area(obj['area_cm2'])}",
                f"  • Centro:     ({obj['center'][0]}, {obj['center'][1]}) px",
                ""
            ])
        elif obj['shape'] == "Polígono":
            s = obj['sides_cm']
            lines.append(f"OBJETO {i}: POLÍGONO ({len(s)} lados)")
            for j, val in enumerate(s):
                label = string.ascii_uppercase[j] if j < 26 else f"L{j+1}"
                lines.append(f"  • Lado {label}: {formato_longitud_from_cm(val)}")
            lines.extend([
                f"  • Perímetro:  {formato_longitud_from_cm(obj['perim_cm'])}",
                f"  • Área:       {formato_area(obj['area_cm2'])}",
                f"  • Centro:     ({obj['center'][0]}, {obj['center'][1]}) px",
                ""
            ])
        else:
            lines.extend([
                f"OBJETO {i}: {obj['shape'].upper()}",
                f"  • Ancho:      {formato_longitud_from_cm(obj.get('w_cm',0.0))}",
                f"  • Alto:       {formato_longitud_from_cm(obj.get('h_cm',0.0))}",
                f"  • Perímetro:  {formato_longitud_from_cm(obj.get('perim_cm',0.0))}",
                f"  • Área:       {formato_area(obj.get('area_cm2',0.0))}",
                f"  • Centro:     ({obj['center'][0]}, {obj['center'][1]}) px",
                ""
            ])
    return "\n".join(lines)

# ------------- Plot y anotaciones -------------
def create_measurement_plot(work_img, results, pixels_per_cm, mode, config, aruco_corners=None):
    img_rgb = cv2.cvtColor(work_img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    fig_width = 10
    fig_height = fig_width * (h / w) + 3
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    ax.imshow(img_rgb)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    cm_step = 2
    x_ticks_px = np.arange(0, w + 1, pixels_per_cm * cm_step)
    y_ticks_px = np.arange(0, h + 1, pixels_per_cm * cm_step)
    x_labels = [f"{i*cm_step:.0f}" for i in range(len(x_ticks_px))]
    y_labels = [f"{i*cm_step:.0f}" for i in range(len(y_ticks_px))]
    ax.set_xticks(x_ticks_px); ax.set_xticklabels(x_labels)
    ax.set_yticks(y_ticks_px); ax.set_yticklabels(y_labels)
    ax.set_xlabel('Ancho (cm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Alto (cm)', fontsize=12, fontweight='bold')
    ax.set_title(f'Mediciones de Objetos - Referencia ArUco ({ARUCO_REAL_SIZE_CM}x{ARUCO_REAL_SIZE_CM} cm)',
                 fontsize=14, fontweight='bold', pad=18)

    if aruco_corners:
        for corner_set in aruco_corners:
            pts = corner_set
            ax.plot(np.append(pts[:, 0], pts[0, 0]),
                    np.append(pts[:, 1], pts[0, 1]),
                    color='lime', linewidth=3, alpha=0.9)
            center_aruco = np.mean(pts, axis=0)
            ax.text(center_aruco[0], center_aruco[1] - 20, 'ArUco\n5x5 cm',
                    fontsize=10, fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lime', alpha=0.8))

    color_map = {
        'Rectángulo': 'red',
        'Cuadrado': 'blue',
        'Círculo': 'green',
        'Triángulo': 'orange',
        'Polígono': 'purple'
    }

    for i, obj in enumerate(results, 1):
        color = color_map.get(obj['shape'], 'gray')

        if obj['shape'] == 'Círculo' and 'radius_px' in obj:
            r_px = obj['radius_px']
            circ = Circle(obj['center'], radius=r_px,
                          fill=False, linewidth=2.5, alpha=0.95, linestyle='-',
                          edgecolor=color)
            ax.add_patch(circ)
            ax.plot(obj['center'][0], obj['center'][1], 'o', color='white', markersize=8,
                    markeredgecolor=color, markeredgewidth=2)
            perim_cm = obj['perim_cm']; radio_cm = obj['radius_cm']
            label_text = (f"Perímetro: {formato_longitud_from_cm(perim_cm)}\n"
                          f"Radio:     {formato_longitud_from_cm(radio_cm)}")
            ax.text(obj['center'][0], obj['center'][1] + 30, label_text,
                    fontsize=10, fontweight='bold', ha='center', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.85, edgecolor='white'),
                    color='white')
            continue  # No plotear líneas de lados para círculo

        elif obj['shape'] == 'Triángulo':
            verts = np.array(obj['vertices_px'])
            ax.plot(np.append(verts[:,0], verts[0,0]), np.append(verts[:,1], verts[0,1]),
                    color=color, linewidth=2.5, alpha=0.95)
            ax.scatter(verts[:,0], verts[:,1], s=40, c='white', edgecolors=color, linewidths=1.5, zorder=5)
            ax.text(obj['center'][0], obj['center'][1], str(i), fontsize=10, fontweight='bold',
                    ha='center', va='center', color='black')
            v0 = verts[0]; v1 = verts[1]; v2 = verts[2]
            side_pts = [ (v0, v1), (v1, v2), (v2, v0) ]
            labels = ['A', 'B', 'C']
            for (p_start, p_end), lab, scm in zip(side_pts, labels, obj['sides_cm']):
                ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], color='yellow', linewidth=3, alpha=0.9)
                midx = (p_start[0] + p_end[0]) / 2.0; midy = (p_start[1] + p_end[1]) / 2.0
                dx = p_end[0] - p_start[0]; dy = p_end[1] - p_start[1]
                length = math.hypot(dx, dy) + 1e-8
                nx = -dy / length; ny = dx / length
                offset = 12
                tx = midx + nx * offset; ty = midy + ny * offset
                text = f"{lab}: {formato_longitud_from_cm(scm)}"
                ax.text(tx, ty, text, fontsize=9, fontweight='bold', ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.25", facecolor='orange', alpha=0.9, edgecolor='white'),
                        color='white')

        elif obj['shape'] in ('Rectángulo', 'Cuadrado'):
            if 'rect' in obj:
                box_points = cv2.boxPoints(obj['rect'])
                box_points = np.intp(box_points)
                ax.plot(np.append(box_points[:, 0], box_points[0, 0]),
                        np.append(box_points[:, 1], box_points[0, 1]),
                        color=color, linewidth=2.5, alpha=0.95)  # Aumentar linewidth para claridad
            ax.plot(obj['center'][0], obj['center'][1], 'o', color='white', markersize=8,
                    markeredgecolor=color, markeredgewidth=2)
            dimensions_text = f"Ancho: {formato_longitud_from_cm(obj['w_cm'])}\nAlto: {formato_longitud_from_cm(obj['h_cm'])}"
            ax.text(obj['center'][0], obj['center'][1] + 30, dimensions_text,
                    fontsize=10, fontweight='bold', ha='center', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.85, edgecolor='white'),
                    color='white')
            # No plotear lados superpuestos; solo el box

        elif obj['shape'] == 'Polígono':
            verts = np.array(obj['vertices_px'])
            n = len(verts)
            ax.plot(np.append(verts[:,0], verts[0,0]), np.append(verts[:,1], verts[0,1]),
                    color=color, linewidth=2.5, alpha=0.95)
            ax.scatter(verts[:,0], verts[:,1], s=30, c='white', edgecolors=color, linewidths=1.2, zorder=5)
            ax.text(obj['center'][0], obj['center'][1], str(i), fontsize=10, fontweight='bold',
                    ha='center', va='center', color='black')
            letters = list(string.ascii_uppercase)
            for idx in range(n):
                p_start = verts[idx]; p_end = verts[(idx+1) % n]
                ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], color='yellow', linewidth=3, alpha=0.9)
                midx = (p_start[0] + p_end[0])/2.0; midy = (p_start[1] + p_end[1])/2.0
                dx = p_end[0] - p_start[0]; dy = p_end[1] - p_start[1]
                length = math.hypot(dx, dy) + 1e-8
                nx = -dy / length; ny = dx / length
                offset = 12
                tx = midx + nx * offset; ty = midy + ny * offset
                lab = letters[idx] if idx < len(letters) else f"L{idx+1}"
                scm = obj['sides_cm'][idx]
                text = f"{lab}: {formato_longitud_from_cm(scm)}"
                ax.text(tx, ty, text, fontsize=9, fontweight='bold', ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.25", facecolor='orange', alpha=0.9, edgecolor='white'),
                        color='white')

        else:
            contour_pts = obj['contour'].reshape(-1, 2)
            ax.plot(np.append(contour_pts[:, 0], contour_pts[0, 0]),
                    np.append(contour_pts[:, 1], contour_pts[0, 1]),
                    color=color, linewidth=2, alpha=0.9)

    ax.grid(True, alpha=0.3)
    info_text = create_detailed_info_text(results, mode, pixels_per_cm)
    plt.subplots_adjust(bottom=0.22)
    fig.text(0.05, 0.02, info_text, fontsize=9, fontfamily='monospace', va='bottom')
    return fig

# ----------------- main -----------------
def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        sys.exit(f"❌ [ERROR] No se pudo leer la imagen: {IMAGE_PATH}")
    h_img, w_img = img.shape[:2]

    config = {
        'ARUCO_REAL_SIZE_CM': ARUCO_REAL_SIZE_CM,
        'MIN_CONTOUR_AREA_PX': 800,
        'ABS_MIN_AREA_PX': 2000,
        'BLUR_KERNEL': 5,
        'MORPH_KERNEL': 3,
        'APPROX_EPS_FACTOR': 0.03,       # Aumentado para mejor aproximación (menos vértices)
        'CIRCULARITY_THRESHOLD': 0.8,   # Bajado para detectar círculos noisy
        'MIN_CIRCULARITY': 0.03,
        'MIN_SOLIDITY': 0.8,             # Aumentado para contornos más limpios
        'MAX_ASPECT_RATIO': 5.0,
        'MIN_AREA_CM2': 2.0,
        'MAX_AREA_CM2': 200.0,
        'MAX_VERTICES': 30
    }

    print(f"[INFO] Resolución: {w_img}x{h_img}")
    print(f"[INFO] Área mínima contorno (px): {config['MIN_CONTOUR_AREA_PX']}")

    work_img, pixels_per_cm, mode, aruco_corners = detectar_referencia(img, config)
    if pixels_per_cm is None or pixels_per_cm <= 0:
        sys.exit("❌ [ERROR] Escala inválida detectada (pixels_per_cm).")

    results = detectar_objetos(work_img, pixels_per_cm, config, mode, aruco_corners)
    print(f"✅ [INFO] {len(results)} objetos detectados y medidos.")

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