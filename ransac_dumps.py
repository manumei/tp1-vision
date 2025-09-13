# Aplicar RANSAC mejorado con los matches optimizados finales
print("=== APLICANDO RANSAC MEJORADO A MATCHES OPTIMIZADOS ===")

# Obtener matches finales con la mejor estrategia
matches_final_10 = match_features(desc1_final, desc0_final, method="sift", 
                                ratio_thresh=0.75, cross_check=False)
matches_final_12 = match_features(desc1_final, desc2_final, method="sift", 
                                ratio_thresh=0.75, cross_check=False)

print(f"Matches disponibles:")
print(f"  udesa1-udesa0: {len(matches_final_10)} matches")
print(f"  udesa1-udesa2: {len(matches_final_12)} matches")


# Aplicar RANSAC mejorado a udesa1-udesa0
print("\n=== RANSAC MEJORADO udesa1 → udesa0 ===")
if len(matches_final_10) >= 4:
    H_10_improved, inliers_10_improved, stats_10 = improved_ransac_homography(
        matches_final_10, kp1_final, kp0_final, T=1000, threshold=3.0)
    print(f"\nHomografía final mejorada:")
    print(H_10_improved)
else:
    print(f"Error: Se necesitan al menos 4 matches, pero solo hay {len(matches_final_10)}")
    H_10_improved, inliers_10_improved, stats_10 = None, [], {}

# Aplicar RANSAC mejorado a udesa1-udesa2
print("\n=== RANSAC MEJORADO udesa1 → udesa2 ===")
if len(matches_final_12) >= 4:
    H_12_improved, inliers_12_improved, stats_12 = improved_ransac_homography(
        matches_final_12, kp1_final, kp2_final, T=1000, threshold=3.0)
    print(f"\nHomografía final mejorada:")
    print(H_12_improved)
else:
    print(f"Error: Se necesitan al menos 4 matches, pero solo hay {len(matches_final_12)}")
    H_12_improved, inliers_12_improved, stats_12 = None, [], {} 
    
# Probar RANSAC con los matches optimizados finales
print("=== APLICANDO RANSAC A MATCHES OPTIMIZADOS ===")

# Obtener matches finales con la mejor estrategia
matches_final_10 = match_features(desc1_final, desc0_final, method="sift", 
                                 ratio_thresh=0.75, cross_check=False)
matches_final_12 = match_features(desc1_final, desc2_final, method="sift", 
                                 ratio_thresh=0.75, cross_check=False)

print(f"Matches disponibles:")
print(f"  udesa1-udesa0: {len(matches_final_10)} matches")
print(f"  udesa1-udesa2: {len(matches_final_12)} matches")

# Aplicar RANSAC a udesa1-udesa0
print("\n=== RANSAC udesa1 → udesa0 ===")
if len(matches_final_10) >= 4:
    H_10, inliers_10 = ransac_homography(matches_final_10, kp1_final, kp0_final, 
                                         T=1000, threshold=3.0)
    print(f"Homografía encontrada:")
    print(H_10)
else:
    print(f"Error: Se necesitan al menos 4 matches, pero solo hay {len(matches_final_10)}")
    H_10, inliers_10 = None, []

# Aplicar RANSAC a udesa1-udesa2
print("\n=== RANSAC udesa1 → udesa2 ===")
if len(matches_final_12) >= 4:
    H_12, inliers_12 = ransac_homography(matches_final_12, kp1_final, kp2_final, 
                                         T=1000, threshold=3.0)
    print(f"Homografía encontrada:")
    print(H_12)
else:
    print(f"Error: Se necesitan al menos 4 matches, pero solo hay {len(matches_final_12)}")
    H_12, inliers_12 = None, [] 










#entre lejos0 y lejos1
if lejos0 is not None and lejos1 is not None:
    kp_best_cerca0, desc_best_cerca0 = detector_best.detectAndCompute(cv2.cvtColor(cerca0, cv2.COLOR_BGR2GRAY), None)
    kp_final_cerca0, desc_final_cerca0, _ = apply_anms_to_sift_features(kp_best_cerca0, desc_best_cerca0, N=300)

    kp_best_cerca1, desc_best_cerca1 = detector_best.detectAndCompute(cv2.cvtColor(cerca1, cv2.COLOR_BGR2GRAY), None)
    kp_final_cerca1, desc_final_cerca1, _ = apply_anms_to_sift_features(kp_best_cerca1, desc_best_cerca1, N=300)

    matches_final_cerca01 = match_features(desc_final_cerca0, desc_final_cerca1, method="sift", 
                                     ratio_thresh=0.75, cross_check=False)

    H_cerca01_improved, inliers_cerca01_improved, stats_cerca0 = ransac_hom0(
            matches_final_cerca01, kp_final_cerca0, kp_final_cerca1, T=1000, threshold=3.0)
    if H_cerca01_improved is not None and H_cerca01_improved.shape != (3,3):
        H_cerca01_improved = np.array(H_cerca01_improved).reshape(3,3)
else:
    print("Error: cerca0 or cerca1 image not loaded.")

#entre cerca1 y cerca2
if cerca1 is not None and cerca2 is not None:
    kp_best_cerca2, desc_best_cerca2 = detector_best.detectAndCompute(cv2.cvtColor(cerca2, cv2.COLOR_BGR2GRAY), None)
    kp_final_cerca2, desc_final_cerca2, _ = apply_anms_to_sift_features(kp_best_cerca2, desc_best_cerca2, N=300)
    matches_final_cerca12 = match_features(desc_final_cerca1, desc_final_cerca2, method="sift", 
                                     ratio_thresh=0.75, cross_check=False)
    H_cerca12_improved, inliers_cerca12_improved, stats_cerca1 = ransac_hom0(
            matches_final_cerca12, kp_final_cerca1, kp_final_cerca2, T=1000, threshold=3.0)
    if H_cerca12_improved is not None and H_cerca12_improved.shape != (3,3):
        H_cerca12_improved = np.array(H_cerca12_improved).reshape(3,3)
else:
    print("Error: cerca1 or cerca2 image not loaded.")










# 3.7 P1 -- Stitching

# Warps (con los puntos manuales del 3.4)
if H_cerca01_improved is not None:
    H_cerca01_improved = np.array(H_cerca01_improved, dtype=np.float32)
    if H_cerca01_improved.shape != (3,3):
        raise ValueError(f"Homografía inválida: {H_cerca01_improved.shape}")
else:
    raise ValueError("No se pudo calcular la homografía entre cerca0 y cerca1")
if H_cerca12_improved is not None:
    H_cerca12_improved = np.array(H_cerca12_improved, dtype=np.float32)
    if H_cerca12_improved.shape != (3,3):
        raise ValueError(f"Homografía inválida: {H_cerca12_improved.shape}")
    
canvas_size_new = (panorama_width, panorama_height)
warp0_cerca = cv2.warpPerspective(cerca0, H_cerca01_improved, canvas_size_new)
warp1_cerca = cv2.warpPerspective(cerca1, H_cerca12_improved, canvas_size_new)

# Masks
mask0_cerca = (warp0_cerca > 0).astype(np.uint8)
mask1_cerca = (warp1_cerca > 0).astype(np.uint8)
# mask2 = (warp2 > 0).astype(np.uint8)

panorama = np.zeros_like(warp1)
for warp, mask in [(warp0_cerca, mask0_cerca), (warp1_cerca, mask1_cerca)]:
    for c in range(3):  # BGR channels
        m = mask[:,:,c] if mask.ndim==3 else mask[:,:,0]
        idx = m > 0 # solo tomar informacion
        
        panorama[:,:,c][idx & (panorama[:,:,c]==0)] = warp[:,:,c][idx & (panorama[:,:,c]==0)] # if panorama empty, copy
        panorama[:,:,c][idx & (panorama[:,:,c]!=0)] = warp[:,:,c][idx & (panorama[:,:,c]!=0)] # si se sobreponen, quedarse con los píxeles del warp (evita ghosting)

plt.figure(figsize=(20,10))
plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
plt.title("Panorama stitched from udesa0, udesa1, udesa2")
plt.axis("off")
plt.show()









# 3.7 P2 -- Blending
num = np.zeros_like(warp0_cerca, dtype=np.float32)
den = np.zeros_like(warp0_cerca[:,:,0], dtype=np.float32)

for warp in [warp0_cerca, warp1_cerca, warp2]:
    mask = (warp.sum(axis=2) > 0).astype(np.uint8)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    W = dist / (dist.max() + 1e-8)

    for c in range(3):
        num[:,:,c] += W * warp[:,:,c]
    den += W

den[den == 0] = 1
panorama = (num / den[:,:,None]).astype(np.uint8)

plt.figure(figsize=(20,10))
plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()










#entre lejos0 y lejos1
if lejos0 is not None and lejos1 is not None:
    kp_best_lejos0, desc_best_lejos0 = detector_best.detectAndCompute(cv2.cvtColor(lejos0, cv2.COLOR_BGR2GRAY), None)
    kp_final_lejos0, desc_final_lejos0, _ = apply_anms_to_sift_features(kp_best_lejos0, desc_best_lejos0, N=300)

    kp_best_lejos1, desc_best_lejos1 = detector_best.detectAndCompute(cv2.cvtColor(lejos1, cv2.COLOR_BGR2GRAY), None)
    kp_final_lejos1, desc_final_lejos1, c_ = apply_anms_to_sift_features(kp_best_lejos1, desc_best_lejos1, N=300)

    matches_final_lejos01 = match_features(desc_final_lejos0, desc_final_lejos1, method="sift", 
                                    ratio_thresh=0.75, cross_check=False)

    H_lejos01_improved, inliers_lejos01_improved, stats_lejos0 = ransac_hom0(
            matches_final_lejos01, kp_final_lejos0, kp_final_lejos1, T=1000, threshold=3.0)
    if H_lejos01_improved is not None and H_lejos01_improved.shape != (3,3):
        H_lejos01_improved = np.array(H_lejos01_improved).reshape(3,3)
else:
    print("Error: cerca0 or cerca1 image not loaded.")












#entre lejos1 y lejos2
if lejos1 is not None and lejos2 is not None:
    kp_best_lejos2, desc_best_lejos2 = detector_best.detectAndCompute(cv2.cvtColor(lejos2, cv2.COLOR_BGR2GRAY), None)
    kp_final_lejos2, desc_final_lejos2, _ = apply_anms_to_sift_features(kp_best_lejos2, desc_best_lejos2, N=300)
    matches_final_lejos12 = match_features(desc_final_lejos1, desc_final_lejos2, method="sift", 
                                     ratio_thresh=0.75, cross_check=False)
    H_lejos12_improved, inliers_lejos12_improved, stats_lejos1 = ransac_hom0(
            matches_final_lejos12, kp_final_lejos1, kp_final_lejos2, T=1000, threshold=3.0)
    if H_lejos12_improved is not None and H_lejos12_improved.shape != (3,3):
        H_lejos12_improved = np.array(H_lejos12_improved).reshape(3,3)
else:
    print("Error: cerca1 or cerca2 image not loaded.")
    
#debug lejos
def debug_matches_and_kp(name1, name2, img1, img2, kp1, desc1, kp2, desc2, matches, max_show=50):
    print(f"DEBUG {name1} ↔ {name2}")
    print(f"  kp {name1}: {0 if kp1 is None else len(kp1)}    desc {name1}: {None if desc1 is None else desc1.shape}")
    print(f"  kp {name2}: {0 if kp2 is None else len(kp2)}    desc {name2}: {None if desc2 is None else desc2.shape}")
    if matches is None:
        print("  matches: None")
        return
    print(f"  matches (good): {len(matches)}")

    # If no matches or <4, stop early
    if len(matches) < 4:
        print("  ¡Pocos matches! No es posible estimar homografía (se requieren al menos 4).")
    else:
        print("  Hay al menos 4 matches — se intentará RANSAC.")

    # Mostrar imagen con matches (hasta max_show)
    try:
        import cv2, matplotlib.pyplot as plt
        draw_matches = matches[:max_show]
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, draw_matches, None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure(figsize=(18,8))
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.title(f"Top {len(draw_matches)} matches {name1} ↔ {name2}")
        plt.axis('off')
        plt.show()
    except Exception as e:
        print("  No se pudo dibujar matches:", e)

# Insertar justo después de que calculás matches_final_cerca01
debug_matches_and_kp("lejos0", "lejos1", lejos0, lejos1,
                     kp_final_lejos0, desc_final_lejos0,
                     kp_final_lejos1, desc_final_lejos1,
                     matches_final_lejos01, max_show=40)










# --- Debug: información rápida antes de RANSAC ---
def debug_matches_and_kp(name1, name2, img1, img2, kp1, desc1, kp2, desc2, matches, max_show=50):
    print(f"DEBUG {name1} ↔ {name2}")
    print(f"  kp {name1}: {0 if kp1 is None else len(kp1)}    desc {name1}: {None if desc1 is None else desc1.shape}")
    print(f"  kp {name2}: {0 if kp2 is None else len(kp2)}    desc {name2}: {None if desc2 is None else desc2.shape}")
    if matches is None:
        print("  matches: None")
        return
    print(f"  matches (good): {len(matches)}")

    # If no matches or <4, stop early
    if len(matches) < 4:
        print("  ¡Pocos matches! No es posible estimar homografía (se requieren al menos 4).")
    else:
        print("  Hay al menos 4 matches — se intentará RANSAC.")

    # Mostrar imagen con matches (hasta max_show)
    try:
        import cv2, matplotlib.pyplot as plt
        draw_matches = matches[:max_show]
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, draw_matches, None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure(figsize=(18,8))
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.title(f"Top {len(draw_matches)} matches {name1} ↔ {name2}")
        plt.axis('off')
        plt.show()
    except Exception as e:
        print("  No se pudo dibujar matches:", e)

# Insertar justo después de que calculás matches_final_cerca01
debug_matches_and_kp("cerca0", "cerca1", cerca0, cerca1,
                     kp_final_cerca0, desc_final_cerca0,
                     kp_final_cerca1, desc_final_cerca1,
                     matches_final_cerca01, max_show=40)
