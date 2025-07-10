import streamlit as st
import cv2
import dlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
import urllib.request

# Page config
st.set_page_config(page_title="Beauty Comparison", layout="wide")
st.title("🌟 Beauty Comparison App")
st.info("📝 NOTE: This is an AI-based comparison for entertainment purposes only.")
st.warning("⚠️ DISCLAIMER: Don't misuse this application.")

# Cache resources
@st.cache_resource
def load_models():
    """Load face detection models"""
    detector = dlib.get_frontal_face_detector()
    
    # Download predictor if not exists
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(predictor_path):
        st.info("Downloading facial landmark model...")
        url = "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat"
        urllib.request.urlretrieve(url, predictor_path)
    
    predictor = dlib.shape_predictor(predictor_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    return detector, predictor, face_cascade, eye_cascade

def preprocess_image(image_path, max_size=600):
    """Load and resize image for processing"""
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    
    return img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def analyze_face_metrics(image_path, detector, predictor, face_cascade, eye_cascade):
    """Comprehensive face analysis"""
    img, gray = preprocess_image(image_path)
    if img is None:
        return {metric: 0 for metric in ['face_shape', 'skin_tone', 'jawline', 'eye_shape', 'eye_color', 'hair_quality']}
    
    # Detect faces
    faces = detector(gray)
    if not faces:
        return {metric: 0 for metric in ['face_shape', 'skin_tone', 'jawline', 'eye_shape', 'eye_color', 'hair_quality']}
    
    face = faces[0]
    landmarks = predictor(gray, face)
    points = np.array([[p.x, p.y] for p in landmarks.parts()])
    
    # 1. Face Shape Score (based on proportions)
    face_width = np.linalg.norm(points[16] - points[0])
    face_height = np.linalg.norm(points[8] - points[27])
    aspect_ratio = face_height / face_width if face_width > 0 else 0
    face_shape_score = min(100, max(0, 100 - abs(aspect_ratio - 1.618) * 50))  # Golden ratio
    
    # 2. Skin Tone Score
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    face_region = img[y1:y2, x1:x2]
    avg_color = np.mean(face_region.reshape(-1, 3), axis=0)
    skin_tone_score = min(100, np.sqrt(np.sum(avg_color**2)) / 4.4)  # Normalized brightness
    
    # 3. Jawline Score (symmetry and definition)
    jaw_points = points[0:17]
    left_jaw = jaw_points[:8]
    right_jaw = jaw_points[9:][::-1]
    
    # Calculate symmetry
    center = points[8]
    left_distances = [np.linalg.norm(p - center) for p in left_jaw]
    right_distances = [np.linalg.norm(p - center) for p in right_jaw]
    symmetry = 100 - np.mean(np.abs(np.array(left_distances) - np.array(right_distances)))
    jawline_score = max(0, min(100, symmetry))
    
    # 4. Eye Shape Score
    left_eye = points[36:42]
    right_eye = points[42:48]
    
    def eye_aspect_ratio(eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)
    
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    avg_ear = (left_ear + right_ear) / 2
    eye_shape_score = min(100, max(0, 100 - abs(avg_ear - 0.3) * 200))  # Ideal EAR ~0.3
    
    # 5. Eye Color Score (using Haar cascades)
    eye_color_score = 50  # Default
    try:
        faces_haar = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces_haar) > 0:
            fx, fy, fw, fh = faces_haar[0]
            face_gray = gray[fy:fy+fh, fx:fx+fw]
            face_color = img[fy:fy+fh, fx:fx+fw]
            eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 3)
            
            if len(eyes) >= 2:
                eye_colors = []
                for ex, ey, ew, eh in eyes[:2]:
                    eye_region = face_color[ey:ey+eh, ex:ex+ew]
                    eye_avg = np.mean(eye_region.reshape(-1, 3), axis=0)
                    eye_colors.append(eye_avg)
                
                if eye_colors:
                    avg_eye_color = np.mean(eye_colors, axis=0)
                    eye_color_score = min(100, np.sqrt(np.sum(avg_eye_color**2)) / 3)
    except:
        pass
    
    # 6. Hair Quality Score (top region analysis)
    try:
        h, w = img.shape[:2]
        hair_region = img[:int(h*0.3), :]  # Top 30%
        hair_avg = np.mean(hair_region.reshape(-1, 3), axis=0)
        hair_brightness = np.sqrt(np.sum(hair_avg**2))
        
        # Edge detection for texture
        hair_gray = cv2.cvtColor(hair_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(hair_gray, 50, 150)
        texture_density = np.sum(edges > 0) / edges.size
        
        hair_quality_score = min(100, (hair_brightness / 4.4) * 0.7 + texture_density * 100 * 0.3)
    except:
        hair_quality_score = 50
    
    return {
        'face_shape': face_shape_score,
        'skin_tone': skin_tone_score,
        'jawline': jawline_score,
        'eye_shape': eye_shape_score,
        'eye_color': eye_color_score,
        'hair_quality': hair_quality_score
    }

def calculate_beauty_score(metrics):
    """Calculate weighted beauty score"""
    weights = {
        'face_shape': 0.25,
        'skin_tone': 0.25,
        'jawline': 0.20,
        'eye_shape': 0.15,
        'eye_color': 0.10,
        'hair_quality': 0.05
    }
    
    return sum(metrics[key] * weights[key] for key in weights)

def add_winner_text(image, is_winner=True):
    """Add winner text to image"""
    if not is_winner:
        return image
    
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    text = "WINNER! 🏆"
    w, h = img_pil.size
    
    # Get text dimensions
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except:
        text_w, text_h = draw.textsize(text, font=font)
    
    # Position at bottom center
    x = (w - text_w) // 2
    y = h - text_h - 20
    
    # Draw background rectangle
    draw.rectangle([x-10, y-5, x+text_w+10, y+text_h+5], fill="gold", outline="black", width=2)
    draw.text((x, y), text, fill="black", font=font)
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# Main app
def main():
    # Load models
    detector, predictor, face_cascade, eye_cascade = load_models()
    
    # File uploaders
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Person 1")
        file1 = st.file_uploader("Upload first image", type=['jpg', 'jpeg', 'png'], key="img1")
    
    with col2:
        st.subheader("👤 Person 2")
        file2 = st.file_uploader("Upload second image", type=['jpg', 'jpeg', 'png'], key="img2")
    
    if file1 and file2:
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp1:
            tmp1.write(file1.getvalue())
            temp_path1 = tmp1.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp2:
            tmp2.write(file2.getvalue())
            temp_path2 = tmp2.name
        
        try:
            # Show progress
            with st.spinner("Analyzing images... Please wait"):
                progress = st.progress(0)
                
                # Analyze first image
                progress.progress(25)
                metrics1 = analyze_face_metrics(temp_path1, detector, predictor, face_cascade, eye_cascade)
                score1 = calculate_beauty_score(metrics1)
                
                # Analyze second image
                progress.progress(75)
                metrics2 = analyze_face_metrics(temp_path2, detector, predictor, face_cascade, eye_cascade)
                score2 = calculate_beauty_score(metrics2)
                
                progress.progress(100)
            
            # Display results
            st.success("Analysis Complete!")
            
            # Results columns
            col3, col4 = st.columns(2)
            
            with col3:
                st.image(file1, caption="Person 1", use_column_width=True)
                st.metric("Beauty Score", f"{score1:.1f}/100", 
                         delta=f"{score1-score2:.1f}" if score1 >= score2 else None)
                
                if st.expander("📊 Detailed Breakdown"):
                    for metric, value in metrics1.items():
                        st.write(f"**{metric.replace('_', ' ').title()}**: {value:.1f}/100")
            
            with col4:
                st.image(file2, caption="Person 2", use_column_width=True)
                st.metric("Beauty Score", f"{score2:.1f}/100",
                         delta=f"{score2-score1:.1f}" if score2 >= score1 else None)
                
                if st.expander("📊 Detailed Breakdown"):
                    for metric, value in metrics2.items():
                        st.write(f"**{metric.replace('_', ' ').title()}**: {value:.1f}/100")
            
            # Winner announcement
            st.markdown("---")
            if abs(score1 - score2) < 1:
                st.success("🤝 **It's a tie! Both are equally beautiful!**")
            elif score1 > score2:
                st.success("🏆 **Person 1 wins!**")
            else:
                st.success("🏆 **Person 2 wins!**")
            
            # Score difference
            diff = abs(score1 - score2)
            if diff > 10:
                st.write(f"**Score difference**: {diff:.1f} points - Clear winner!")
            elif diff > 5:
                st.write(f"**Score difference**: {diff:.1f} points - Close competition!")
            else:
                st.write(f"**Score difference**: {diff:.1f} points - Very close!")
                
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            
        finally:
            # Clean up
            try:
                os.unlink(temp_path1)
                os.unlink(temp_path2)
            except:
                pass
    
    else:
        st.info("👆 Please upload both images to start the comparison")

if __name__ == "__main__":
    main()
