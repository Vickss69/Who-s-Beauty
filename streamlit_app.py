import streamlit as st
import cv2
import dlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
import time
import urllib.request
import traceback

st.set_page_config(page_title="Beauty Score Comparator", layout="wide")

st.info("📝 NOTE: The present comparison is only in the image, it's not in real life.")
st.warning("⚠️ DISCLAIMER: Don't misuse this application.")

# Cache the detector and predictor to avoid reloading on each run
@st.cache_resource
def load_face_detector():
    """Loads the Dlib frontal face detector."""
    return dlib.get_frontal_face_detector()

@st.cache_resource
def load_landmark_predictor():
    """
    Downloads and loads the Dlib facial landmark predictor model.
    The model is cached to avoid re-downloading.
    """
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(predictor_path):
        st.warning("Downloading facial landmark predictor model. This may take a moment...")
        model_url = "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat"
        urllib.request.urlretrieve(model_url, predictor_path)
        st.success("Download complete!")
    return dlib.shape_predictor(predictor_path)

@st.cache_resource
def load_haar_cascades():
    """Loads OpenCV's Haar Cascade classifiers for face and eye detection."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    return face_cascade, eye_cascade

# Set page title and description
st.title("Know who's more Beautiful")
st.write("Upload two images to compare and see which one scores higher!")

# Preload the detectors and predictors when the app starts
detector = load_face_detector()
predictor = load_landmark_predictor()
face_cascade, eye_cascade = load_haar_cascades()

def calculate_face_shape(landmarks):
    # Extract key facial landmarks
    jaw = landmarks[:17]
    forehead_width = np.linalg.norm(landmarks[16] - landmarks[0])
    face_height = np.linalg.norm(landmarks[8] - landmarks[27])
    cheek_width = np.linalg.norm(landmarks[13] - landmarks[3])

    # Estimate forehead region
    forehead_height = (landmarks[19][1] + landmarks[24][1]) // 2 - landmarks[27][1]
    total_face_height = face_height + forehead_height

    # Ratios for classification
    aspect_ratio = total_face_height / forehead_width
    cheek_to_jaw_ratio = cheek_width / forehead_width

    # Classification based on geometric ratios - returns a raw score
    if aspect_ratio < 1.3 and cheek_to_jaw_ratio < 0.9: return 8   # Square
    elif aspect_ratio >= 1.3 and cheek_to_jaw_ratio < 0.9: return 18  # Oval
    elif cheek_to_jaw_ratio >= 1.0: return 11  # Round
    elif cheek_to_jaw_ratio < 0.8 and aspect_ratio > 1.5: return 15  # Heart
    elif landmarks[13][0] - landmarks[3][0] > cheek_width / 2 and landmarks[8][1] > landmarks[13][1]: return 26  # Diamond
    elif cheek_width < forehead_width and aspect_ratio > 1.4: return 0  # Default case or less defined
    else: return 22  # Oblong

def detect_face_shape(image_path, detector=detector, predictor=predictor):
    try:
        image = cv2.imread(image_path)
        if image is None: return 0
        height, width = image.shape[:2]
        max_dim = 800
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if not faces: return 0

        for face in faces:
            landmarks = predictor(gray, face)
            landmarks_points = np.array([[p.x, p.y] for p in landmarks.parts()])
            forehead_left = landmarks_points[19] - np.array([0, 30])
            forehead_right = landmarks_points[24] - np.array([0, 30])
            extended_landmarks = np.vstack([landmarks_points, forehead_left, forehead_right])
            face_shape_score = calculate_face_shape(extended_landmarks)
            # Normalize the score to a 0-100 scale
            return (face_shape_score / 26) * 100
        return 0
    except Exception as e:
        # st.error(f"Error in face shape detection: {e}")
        return 0

def get_average_skin_color(image_path, detector=detector):
    try:
        image = cv2.imread(image_path)
        if image is None: return 0
        height, width = image.shape[:2]
        max_dim = 800
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = detector(image_rgb)
        if len(faces) == 0: return 0

        skin_pixels = []
        for face in faces:
            x1, y1, x2, y2 = (face.left(), face.top(), face.right(), face.bottom())
            roi = image[y1:y2, x1:x2]
            h, w, _ = roi.shape
            step = max(1, min(h, w) // 20)
            for i in range(0, roi.shape[0], step):
                for j in range(0, roi.shape[1], step):
                    skin_pixels.append(roi[i, j])
        if not skin_pixels: return 0
        
        avg_bgr = np.mean(np.array(skin_pixels), axis=0)
        a, b, c = avg_bgr
        S = (np.sqrt(a**2 + b**2 + c**2))
        return S
    except Exception as e:
        # st.error(f"Error in skin color detection: {e}")
        return 0

def rate_jawline(image_path, detector=detector, predictor=predictor):
    try:
        img = cv2.imread(image_path)
        if img is None: return 0
        height, width = img.shape[:2]
        max_dim = 800
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            img = cv2.resize(img, (int(width * scale), int(height * scale)))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if len(faces) == 0: return 0

        landmarks = predictor(gray, faces[0])
        jawline = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(0, 17)])
        left_jawline, right_jawline = jawline[:9], jawline[8:]
        center_point = jawline[8]
        left_distances = [np.linalg.norm(pt - center_point) for pt in left_jawline]
        right_distances = [np.linalg.norm(pt - center_point) for pt in right_jawline[::-1]]
        symmetry_score = 100 - np.mean(np.abs(np.array(left_distances) - np.array(right_distances)))
        
        def calculate_angle(a, b, c):
            ba = a - b; bc = c - b
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
            
        angles = [calculate_angle(jawline[i - 1], jawline[i], jawline[i + 1]) for i in range(1, len(jawline) - 1)]
        sharpness_score = 100 - np.mean(np.abs(np.array(angles) - 120))
        return 0.6 * np.clip(symmetry_score, 0, 100) + 0.4 * np.clip(sharpness_score, 0, 100)
    except Exception as e:
        # st.error(f"Error in jawline detection: {e}")
        return 0

def calculate_eye_shape(eye_landmarks):
    width = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
    if width == 0: return 12 # Prevent division by zero
    height = (np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5])) +
              np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))) / 2
    aspect_ratio = height / width
    if aspect_ratio < 0.25: return 38 # Almond
    elif 0.25 <= aspect_ratio <= 0.35: return 28 # Upturned/Downturned
    elif aspect_ratio > 0.35: return 22 # Round
    else: return 12

def detect_eyes_shape(image_path, detector=detector, predictor=predictor):
    try:
        img = cv2.imread(image_path)
        if img is None: return 0
        height, width = img.shape[:2]
        max_dim = 800
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            img = cv2.resize(img, (int(width * scale), int(height * scale)))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if not faces: return 0

        for face in faces:
            landmarks = predictor(gray, face)
            left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
            left_eye_shape = calculate_eye_shape(left_eye)
            right_eye_shape = calculate_eye_shape(right_eye)
            avg_shape_score = (left_eye_shape + right_eye_shape) / 2
            # Normalize the score out of max possible score (38) to 0-100 scale
            return (avg_shape_score / 38) * 100
        return 0
    except Exception as e:
        # st.error(f"Error in eye shape detection: {e}")
        return 0

def classify_eye_color(rgb_values):
    r, g, b = rgb_values
    if r > 100 and g < 70 and b < 40: return 5  # Brown
    elif r > 140 and g > 100 and b < 60: return 19  # Amber
    elif r < 100 and g < 100 and b > 120: return 29  # Blue
    elif r < 100 and g > 120 and b < 100: return 14  # Green
    elif r > 100 and g > 80 and b < 60: return 9   # Hazel
    elif r < 100 and g < 100 and b < 80: return 24  # Gray
    else: return 15  # Other/Indeterminate

def detect_eye_colors(image_path, face_cascade=face_cascade, eye_cascade=eye_cascade):
    try:
        image = cv2.imread(image_path)
        if image is None: return 0
        height, width = image.shape[:2]
        max_dim = 800
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0: return 0
        
        (x, y, w, h) = faces[0]
        face_region = image[y:y+h, x:x+w]
        face_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 3)
        if len(eyes) < 2: return 0
        
        eyes = sorted(eyes, key=lambda e: e[0])[:2]
        eye_colors = []
        for (ex,ey,ew,eh) in eyes:
            eye_region = face_region[ey:ey+eh, ex:ex+ew]
            if eye_region.size > 0:
                avg_rgb = np.mean(eye_region, axis=(0, 1))
                eye_colors.append(classify_eye_color(avg_rgb))
        
        if not eye_colors: return 0
        avg_color_score = np.mean(eye_colors)
        # Normalize score out of max possible (29) to 0-100 scale
        return (avg_color_score / 29) * 100
    except Exception as e:
        # st.error(f"Error in eye color detection: {e}")
        return 0

def calculate_final_hair_score(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None: return 0
        height, width = image.shape[:2]
        max_dim = 800
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))
        
        # Color score part
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hair_region = image_rgb[:int(height * 0.4), :]
        if hair_region.size == 0: return 0
        
        avg_rgb = np.mean(hair_region, axis=(0,1))
        a, b, c = avg_rgb
        color_magnitude = np.sqrt(a**2 + b**2 + c**2)
        max_magnitude = np.sqrt(3 * 255**2)
        color_score = ((max_magnitude - color_magnitude) / max_magnitude) * 100
        
        # Density part
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)
        hair_mask = np.zeros_like(edges)
        mask_height = int(gray_image.shape[0] * 0.4)
        hair_mask[:mask_height, :] = 255
        total_pixels = np.sum(hair_mask > 0)
        if total_pixels == 0: density_score = 0
        else:
            hair_pixels = np.sum(cv2.bitwise_and(edges, hair_mask) > 0)
            density_score = (hair_pixels / total_pixels) * 100
            
        return (color_score + density_score) / 2
    except Exception as e:
        # st.error(f"Error in final hair score calculation: {e}")
        return 0

def mark_winner(image_path):
    try:
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        text = "WINNER"
        try:
            font = ImageFont.truetype("arial.ttf", 60)
        except IOError:
            font = ImageFont.load_default()

        try: bbox = draw.textbbox((0, 0), text, font=font)
        except AttributeError: text_width, text_height = draw.textsize(text, font=font)
        else: text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            
        pos_x = (image.width - text_width) / 2
        pos_y = image.height - text_height - 30
        
        # Draw a semi-transparent red banner
        draw.rectangle(
            (0, pos_y - 10, image.width, pos_y + text_height + 10),
            fill=(255, 0, 0, 128) # RGBA with transparency
        )
        draw.text((pos_x, pos_y), text, fill="white", font=font)
        return image
    except Exception as e:
        # st.error(f"Error marking winner: {e}")
        return Image.open(image_path) # Return original image on error

def analyze_image(image_path, progress_callback=None):
    metrics = {}
    if progress_callback: progress_callback(0, "Detecting face...")
    metrics['face_shape'] = detect_face_shape(image_path)
    if metrics['face_shape'] == 0:
        if progress_callback: progress_callback(100, "No face detected!")
        return {key: 0 for key in ['face_shape', 'skin_score', 'jawline', 'eye_shape', 'eye_color', 'hair_score', 'final_score']}
    
    if progress_callback: progress_callback(15, "Analyzing skin color...")
    skin_color_raw = get_average_skin_color(image_path)
    metrics['skin_score'] = (1.5 * 100 * skin_color_raw / (256 * (3**0.5)))
    
    if progress_callback: progress_callback(30, "Analyzing jawline...")
    metrics['jawline'] = rate_jawline(image_path)
    
    if progress_callback: progress_callback(45, "Analyzing eye shape...")
    metrics['eye_shape'] = detect_eyes_shape(image_path)
    
    if progress_callback: progress_callback(60, "Analyzing eye color...")
    metrics['eye_color'] = detect_eye_colors(image_path)
    
    if progress_callback: progress_callback(75, "Analyzing hair...")
    metrics['hair_score'] = calculate_final_hair_score(image_path)
    
    metrics['final_score'] = (
        metrics['face_shape'] * 0.25 + 
        metrics['skin_score'] * 0.20 + 
        metrics['jawline']    * 0.20 + 
        metrics['eye_shape']  * 0.15 + 
        metrics['eye_color']  * 0.10 + 
        metrics['hair_score'] * 0.10
    )
    if progress_callback: progress_callback(100, "Analysis complete!")
    return metrics

# --- Streamlit UI ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("Image 1")
    uploaded_file1 = st.file_uploader("Choose first image", type=['jpg', 'jpeg', 'png'], key="file1")

with col2:
    st.subheader("Image 2")
    uploaded_file2 = st.file_uploader("Choose second image", type=['jpg', 'jpeg', 'png'], key="file2")

if uploaded_file1 is not None and uploaded_file2 is not None:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    temp_file1_path, temp_file2_path = None, None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file1:
            temp_file1.write(uploaded_file1.getvalue())
            temp_file1_path = temp_file1.name
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file2:
            temp_file2.write(uploaded_file2.getvalue())
            temp_file2_path = temp_file2.name

        def update_progress(percent, message=""):
            progress_bar.progress(int(percent))
            if message: status_text.text(message)
        
        metrics1 = analyze_image(temp_file1_path, lambda p, m: update_progress(p / 2, f"Image 1: {m}"))
        metrics2 = analyze_image(temp_file2_path, lambda p, m: update_progress(50 + p / 2, f"Image 2: {m}"))
        
        status_text.text("Comparison complete!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        
        s1, s2 = metrics1['final_score'], metrics2['final_score']
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.image(uploaded_file1, caption=f"Image 1 | Score: {s1:.2f}")
            with st.expander("Detailed Scores for Image 1"):
                st.write(f"Face Shape: {metrics1['face_shape']:.2f}")
                st.write(f"Skin Score: {metrics1['skin_score']:.2f}")
                st.write(f"Jawline: {metrics1['jawline']:.2f}")
                st.write(f"Eye Shape: {metrics1['eye_shape']:.2f}")
                st.write(f"Eye Color: {metrics1['eye_color']:.2f}")
                st.write(f"Hair Score: {metrics1['hair_score']:.2f}")

        with res_col2:
            st.image(uploaded_file2, caption=f"Image 2 | Score: {s2:.2f}")
            with st.expander("Detailed Scores for Image 2"):
                st.write(f"Face Shape: {metrics2['face_shape']:.2f}")
                st.write(f"Skin Score: {metrics2['skin_score']:.2f}")
                st.write(f"Jawline: {metrics2['jawline']:.2f}")
                st.write(f"Eye Shape: {metrics2['eye_shape']:.2f}")
                st.write(f"Eye Color: {metrics2['eye_color']:.2f}")
                st.write(f"Hair Score: {metrics2['hair_score']:.2f}")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("🏆 The Winner Is...")
        
        if s1 == 0 and s2 == 0:
            st.error("Could not detect faces in either image. Unable to determine a winner.")
        elif s1 >= s2:
            winner_pil = mark_winner(temp_file1_path)
            st.image(winner_pil, caption="Image 1 Wins!", use_column_width=True)
        else:
            winner_pil = mark_winner(temp_file2_path)
            st.image(winner_pil, caption="Image 2 Wins!", use_column_width=True)

    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
        st.error(traceback.format_exc())
    
    finally:
        # Clean up the temporary files
        if temp_file1_path and os.path.exists(temp_file1_path): os.unlink(temp_file1_path)
        if temp_file2_path and os.path.exists(temp_file2_path): os.unlink(temp_file2_path)
else:
    st.info("Please upload both images to begin the comparison.")
