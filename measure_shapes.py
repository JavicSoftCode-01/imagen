"""
measure_shapes_enhanced_v2_debug.py
Versión con debug para visualizar el proceso de detección
"""
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
from datetime import datetime

# --- CONFIG ---
IMAGE_PATH = "cercatarjeta.jpg"
OUTPUT_PATH = IMAGE_PATH + ".png"
ARUCO_REAL_SIZE_CM = 5.0

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

def detectar_forma(contour, approx_eps_factor):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, approx_eps_factor * peri, True)
    v = len(approx)
    
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
        area = cv2.contourArea(contour)
        if peri > 0:
            circularity = 4 * math.pi * area / (peri * peri)
            if circularity > 0.75:
                return "Círculo", approx
    
    return "Polígono", approx

def detectar_referencia(image, config):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    aruco_params = cv2.aruco.DetectorParameters()
    corners_list, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    if ids is None or len(ids) == 0:
        sys.exit("❌ [ERROR] No se detectó el marcador ArUco en la imagen.")
    
    c = corners_list[0].reshape((4,2))
    lado1 = np.linalg.norm(c[1] - c[0])
    lado2 = np.linalg.norm(c[2] - c[1])
    lado3 = np.linalg.norm(c[3] - c[2])
    lado4 = np.linalg.norm(c[0] - c[3])
    
    avg_pixels = (lado1 + lado2 + lado3 + lado4) / 4.0
    pixels_per_cm = avg_pixels / config['ARUCO_REAL_SIZE_CM']
    
    print(f"✅ [INFO] ArUco detectado. Escala: {pixels_per_cm:.2f} px/cm")
    print(f"    Lados ArUco: {lado1:.1f}, {lado2:.1f}, {lado3:.1f}, {lado4:.1f} px")
    print(f"    Promedio: {avg_pixels:.1f}px = {config['ARUCO_REAL_SIZE_CM']}cm")
    
    return image.copy(), pixels_per_cm, 'aruco', corners_list

def detectar_objetos(work_img, pixels_per_cm, config, mode, aruco_corners=None):
    """Versión con debug visual del proceso"""
    h_img, w_img = work_img.shape[:2]
    gray_w = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
    
    print(f"[DEBUG] Procesando imagen {w_img}x{h_img}")
    
    # Múltiples métodos de detección
    methods = []
    
    # Método 1: Threshold adaptativo
    adaptive_th = cv2.adaptiveThreshold(gray_w, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 21, 10)
    methods.append(("Adaptive Threshold", adaptive_th))
    
    # Método 2: Otsu
    blur = cv2.GaussianBlur(gray_w, (5, 5), 0)
    _, otsu_th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    methods.append(("Otsu", otsu_th))
    
    # Método 3: Canny
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    canny_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    methods.append(("Canny + Close", canny_closed))
    
    # Crear máscara de exclusión del ArUco
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
    
    # Probar cada método y encontrar contornos
    all_results = []
    best_method = None
    max_objects = 0
    
    for method_name, binary_img in methods:
        # Limpiar imagen binaria
        cleaned = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Aplicar máscara de exclusión
        if mode == 'aruco':
            cleaned = cv2.bitwise_and(cleaned, cv2.bitwise_not(exclude_mask))
        
        # Encontrar contornos
        cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar contornos con criterios mejorados
        valid_objects = []
        for c in cnts:
            area_px = cv2.contourArea(c)
            
            # Filtros de tamaño más restrictivos
            if area_px < config['MIN_CONTOUR_AREA_PX']:
                continue
            if area_px > (w_img * h_img * 0.1):  # Máximo 10% de la imagen
                continue
            if area_px < 2000:  # Mínimo absoluto más alto
                continue
                
            # Filtros de forma
            rect = cv2.minAreaRect(c)
            (_, (w_px, h_px), _) = rect
            if w_px == 0 or h_px == 0:
                continue
            
            # Aspect ratio más restrictivo para objetos reales
            aspect_ratio = max(w_px, h_px) / min(w_px, h_px)
            if aspect_ratio > 5:  # Máximo 5:1 de proporción
                continue
            
            # Filtro de circularidad para eliminar formas muy irregulares
            peri = cv2.arcLength(c, True)
            if peri > 0:
                circularity = 4 * math.pi * area_px / (peri * peri)
                if circularity < 0.1:  # Muy irregular
                    continue
            
            # Filtro de solidez (área del contorno vs área del hull convexo)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area_px / hull_area
                if solidity < 0.5:  # Muy cóncavo/fragmentado
                    continue
            
            # Filtro de tamaño mínimo en centímetros (evitar objetos muy pequeños)
            area_cm2 = area_px / (pixels_per_cm * pixels_per_cm)
            if area_cm2 < 2.0:  # Menos de 2 cm²
                continue
            if area_cm2 > 200.0:  # Más de 200 cm² (probablemente fondo)
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
    
    # Procesar objetos encontrados
    results = []
    for i, c in enumerate(valid_contours):
        area_px = cv2.contourArea(c)
        rect = cv2.minAreaRect(c)
        (center_px, (w_px, h_px), angle) = rect
        
        # Asegurar orden correcto
        if h_px > w_px:
            w_px, h_px = h_px, w_px
        
        shape, approx = detectar_forma(c, config['APPROX_EPS_FACTOR'])
        perim_px = cv2.arcLength(c, True)
        
        # Convertir a centímetros
        w_cm = w_px / pixels_per_cm
        h_cm = h_px / pixels_per_cm
        area_cm2 = area_px / (pixels_per_cm * pixels_per_cm)
        perim_cm = perim_px / pixels_per_cm
        
        cx, cy = int(center_px[0]), int(center_px[1])
        
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
    
    print(f"[DEBUG] Mejor método: {method_name} con {len(results)} objetos")
    return results

def create_measurement_plot(work_img, results, pixels_per_cm, mode, config, aruco_corners=None):
    """Crea el plot final con mediciones"""
    img_rgb = cv2.cvtColor(work_img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    fig_width = 12
    fig_height = fig_width * (h / w) + 4
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    ax.imshow(img_rgb)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    
    # Configurar ejes
    cm_step = 2
    x_ticks_px = np.arange(0, w, pixels_per_cm * cm_step)
    y_ticks_px = np.arange(0, h, pixels_per_cm * cm_step)
    x_labels = [f"{i*cm_step:.0f}" for i in range(len(x_ticks_px))]
    y_labels = [f"{i*cm_step:.0f}" for i in range(len(y_ticks_px))]
    
    ax.set_xticks(x_ticks_px)
    ax.set_xticklabels(x_labels)
    ax.set_yticks(y_ticks_px)
    ax.set_yticklabels(y_labels)
    
    ax.set_xlabel('Ancho (cm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Alto (cm)', fontsize=12, fontweight='bold')
    ax.set_title(f'Mediciones de Objetos - Referencia ArUco ({ARUCO_REAL_SIZE_CM}x{ARUCO_REAL_SIZE_CM} cm)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Dibujar ArUco
    if aruco_corners:
        for corner_set in aruco_corners:
            pts = corner_set.reshape(4, 2)
            ax.plot(np.append(pts[:, 0], pts[0, 0]), 
                   np.append(pts[:, 1], pts[0, 1]), 
                   color='lime', linewidth=3, alpha=0.8)
            center_aruco = np.mean(pts, axis=0)
            ax.text(center_aruco[0], center_aruco[1] - 20, 'ArUco\n5x5 cm', 
                   fontsize=10, fontweight='bold', ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lime', alpha=0.7))
    
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
        
        # Dibujar contorno
        ax.plot(np.append(contour_pts[:, 0], contour_pts[0, 0]), 
               np.append(contour_pts[:, 1], contour_pts[0, 1]), 
               color=color, linewidth=2, alpha=0.8)
        
        # Punto central
        ax.plot(obj['center'][0], obj['center'][1], 'o', 
               color='white', markersize=8, 
               markeredgecolor=color, markeredgewidth=2)
        
        # Número del objeto
        ax.text(obj['center'][0], obj['center'][1], str(i), 
               fontsize=10, fontweight='bold', ha='center', va='center', color='black')
        
        # Añadir dimensiones sobre el objeto
        dimensions_text = f"Ancho: {formato_longitud_from_cm(obj['w_cm'])}\nAlto: {formato_longitud_from_cm(obj['h_cm'])}"
        ax.text(obj['center'][0], obj['center'][1] + 30, dimensions_text, 
               fontsize=10, fontweight='bold', ha='center', va='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7, edgecolor='white'),
               color='white')
    
    ax.grid(True, alpha=0.3)
    
    # Información detallada
    info_text = create_detailed_info_text(results, mode, pixels_per_cm)
    plt.subplots_adjust(bottom=0.25)
    fig.text(0.1, 0.02, info_text, fontsize=9, fontfamily='monospace', va='bottom')
    
    return fig

def create_detailed_info_text(results, mode, pixels_per_cm):
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

def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        sys.exit(f"❌ [ERROR] No se pudo leer la imagen: {IMAGE_PATH}")
    
    h_img, w_img = img.shape[:2]
    
    # Configuración más permisiva
    config = {
        'ARUCO_REAL_SIZE_CM': ARUCO_REAL_SIZE_CM,
        'MIN_CONTOUR_AREA_PX': 800,  # Reducido aún más
        'BLUR_KERNEL': 5,
        'MORPH_KERNEL': 3,
        'APPROX_EPS_FACTOR': 0.02
    }
    
    print(f"[INFO] Resolución: {w_img}x{h_img}")
    print(f"[INFO] Área mínima contorno: {config['MIN_CONTOUR_AREA_PX']} px")
    
    # Detectar ArUco
    work_img, pixels_per_cm, mode, aruco_corners = detectar_referencia(img, config)
    
    if pixels_per_cm is None:
        sys.exit("❌ [ERROR] No se pudo establecer referencia de escala.")
    
    # Detectar objetos normalmente
    results = detectar_objetos(work_img, pixels_per_cm, config, mode, aruco_corners)
    
    print(f"✅ [INFO] {len(results)} objetos detectados y medidos.")
    
    # Crear reporte final si hay resultados
    if results:
        fig = create_measurement_plot(work_img, results, pixels_per_cm, mode, config, aruco_corners)
        fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ [INFO] Imagen guardada en: {OUTPUT_PATH}")
    else:
        print("⚠️ [WARN] No se detectaron objetos para medir.")

if __name__ == "__main__":
    main()