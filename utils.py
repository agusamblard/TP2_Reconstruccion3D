import cv2
import numpy as np
import matplotlib.pyplot as plt

def rectify_stereo_pair(left_img, right_img, stereo_maps):
    """
    Rectifica un par estéreo usando mapas pre-computados.
    Args:
        left_img (str): Ruta a la imagen izquierda.
        right_img (str): Ruta a la imagen derecha.
        stereo_maps (dict): Diccionario con los mapas de remapeo para ambas cámaras.
    Returns:
        left_rect (ndarray): Imagen izquierda rectificada.
        right_rect (ndarray): Imagen derecha rectificada.
    """
    left_img  = cv2.imread(left_img)
    right_img = cv2.imread(right_img)
    left_rect = cv2.remap(left_img, 
                         stereo_maps['left_map_x'], 
                         stereo_maps['left_map_y'], 
                         cv2.INTER_LINEAR)
    
    right_rect = cv2.remap(right_img,
                          stereo_maps['right_map_x'],
                          stereo_maps['right_map_y'],
                          cv2.INTER_LINEAR)
    
    return left_rect, right_rect

def draw_epipolar_lines(left_rect, right_rect, num_lines=20):
    """Dibuja líneas horizontales para verificar rectificación"""
    combined = np.hstack([left_rect, right_rect])
    h, w = left_rect.shape[:2]
    
    for y in range(0, h, h//num_lines):
        cv2.line(combined, (0, y), (2*w, y), (0, 255, 0), 3)
    
    return combined

def show_pair_any(idx=14, rectificar=True, num_lines=20, stereo_maps=None):
    left_path, right_path = f"data/captures/left_{idx}.jpg", f"data/captures/right_{idx}.jpg"

    if rectificar:
        L, R = rectify_stereo_pair(left_path, right_path, stereo_maps)
        titulo = "Rectificadas"
    else:
        L, R = cv2.imread(left_path), cv2.imread(right_path)
        titulo = "Originales (sin rectificar)"

    combined = draw_epipolar_lines(L, R, num_lines=num_lines)

    plt.figure(figsize=(18,9))
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.title(f"{titulo} - Par {idx}", fontsize=16)
    plt.axis("off")
    plt.show()

def show_pair_any_budha(idx=14, rectificar=True, num_lines=20, stereo_maps=None):
    left_path, right_path = f"data/stereo_budha_charuco/captures/left_{idx}.jpg", f"data/stereo_budha_charuco/captures/right_{idx}.jpg"

    if rectificar:
        L, R = rectify_stereo_pair(left_path, right_path, stereo_maps)
        titulo = "Rectificadas"
    else:
        L, R = cv2.imread(left_path), cv2.imread(right_path)
        titulo = "Originales (sin rectificar)"

    combined = draw_epipolar_lines(L, R, num_lines=num_lines)

    plt.figure(figsize=(18,9))
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.title(f"{titulo} - Par {idx}", fontsize=16)
    plt.axis("off")
    plt.show()
def compute_depth(disparity_map, f, B, default=1000.0):

    # Crea una copia del mapa de disparidad
    disparity_map = disparity_map.copy()
    
    # Evita divisiones por cero o disparidades negativas (les asignamos el valor default)
    mask_invalid = (disparity_map <= 0)
    
    # Calcula la profundidad con la fórmula Z = f * B / disparidad
    depth_map = np.zeros_like(disparity_map, dtype=np.float32)
    depth_map[~mask_invalid] = (f * B) / disparity_map[~mask_invalid]
    
    # Asigna valor fijo a los puntos donde la disparidad es inválida
    depth_map[mask_invalid] = default
    
    return depth_map