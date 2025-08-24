import cv2
import numpy as np

def derive_graym(impath):
    return cv2.imread(impath, cv2.IMREAD_GRAYSCALE)

def derive_m(img):
    return np.mean(img, axis=2).astype(np.uint8)

def derive_saturation(img, rimg):
    s_img = np.zeros_like(rimg, dtype=np.float32)
    b, g, r = cv2.split(img)
    s1 = np.clip(b + r, 0, 255)
    s2 = 2 * g
    mask = s1 >= s2
    s_img[mask] = 1.5 * np.clip(r[mask] - rimg[mask], 0, 255)
    s_img[~mask] = 1.5 * np.clip(rimg[~mask] - b[~mask], 0, 255)
    return s_img.astype(np.uint8)

def check_pixel_specularity(mimg, simg, m_thresh=0.7, s_thresh=0.2):
    """Threshold lebih ketat untuk specularity kecil & sangat terang."""
    m_max = np.max(mimg) * m_thresh
    s_max = np.max(simg) * s_thresh
    spec_mask = np.zeros_like(mimg, dtype=np.uint8)
    spec_mask[(mimg >= m_max) & (simg <= s_max)] = 255
    return spec_mask

def enlarge_specularity(spec_mask, win_size=(3,3), step_size=1):
    """Gunakan window kecil agar tidak meluber."""
    enlarged_spec = np.copy(spec_mask)
    for r in range(0, spec_mask.shape[0] - win_size[0] + 1, step_size):
        for c in range(0, spec_mask.shape[1] - win_size[1] + 1, step_size):
            win = spec_mask[r:r + win_size[0], c:c + win_size[1]]
            if np.any(win):
                enlarged_spec[r:r + win_size[0], c:c + win_size[1]] = 255
    return enlarged_spec

def remove_specularity(impath, radius=5, m_thresh=0.7, s_thresh=0.2, win_size=(3,3)):
    """Tuned untuk pantulan kecil & terang."""
    img = cv2.imread(impath)
    gray_img = derive_graym(impath)
    rimg = derive_m(img)
    simg = derive_saturation(img, rimg)
    spec_mask = check_pixel_specularity(rimg, simg, m_thresh, s_thresh)
    enlarged_spec = enlarge_specularity(spec_mask, win_size=win_size)
    telea = cv2.inpaint(img, enlarged_spec, radius, cv2.INPAINT_TELEA)
    ns = cv2.inpaint(img, enlarged_spec, radius, cv2.INPAINT_NS)
    return telea, ns

# Example usage
impath = '../images/normal.jpg'
telea, ns = remove_specularity(
    impath='../images/normal.jpg',
    radius=3,          # radius kecil
    m_thresh=0.95,      # intensitas sangat tinggi
    s_thresh=0.15,      # saturasi rendah
    win_size=(3,3)      # window kecil agar tidak melebar
)
cv2.imwrite('Impainted_telea.png', telea)
cv2.imwrite('Impainted_ns.png', ns)
