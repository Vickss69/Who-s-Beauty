import streamlit as st
import cv2
import dlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
import time

st.info("📝 NOTE: The present comparison is only in the image, it's not in real life.")
st.warning("⚠️ DISCLAIMER: Don't misuse this application.")

# Cache the detector and predictor to avoid reloading
@st.cache_resource
def load_face_detector():
    return dlib.get_frontal_face_detector()

@st.cache_resource
def load_landmark_predictor():
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(predictor_path):
        st.warning("Downloading facial landmark predictor model. This may take a moment...")
        import urllib.request
        model_url = "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat"
        urllib.request.urlretrieve(model_url, predictor_path)
        st.success("Download complete!")
    return dlib.shape_predictor(predictor_path)

@st.cache_resource
def load_haar_cascades():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    return face_cascade, eye_cascade

# Set page title and description
st.title("Know who's more Beautiful")
st.write("Upload two images to compare and see which one scores higher!")

# Preload the detectors and predictors
detector = load_face_detector()
predictor = load_landmark_predictor()
face_cascade, eye_cascade = load_haar_cascades()

def calculate_face_shape(landmarks):
    # Extract key facial landmarks
    jaw = landmarks[:17]  # Jawline points
    forehead_width = np.linalg.norm(landmarks[16] - landmarks[0])  # Width across forehead
    face_height = np.linalg.norm(landmarks[8] - landmarks[27])  # From chin to middle forehead
    cheek_width = np.linalg.norm(landmarks[13] - landmarks[3])  # Widest cheekbone distance

    # Estimate forehead region
    forehead_height = (landmarks[19][1] + landmarks[24][1]) // 2 - landmarks[27][1]
    total_face_height = face_height + forehead_height

    # Ratios for classification
    aspect_ratio = total_face_height / forehead_width
    cheek_to_jaw_ratio = cheek_width / forehead_width

    # Classification based on geometric ratios
    if aspect_ratio < 1.3 and cheek_to_jaw_ratio < 0.9:
        return 8
    elif aspect_ratio >= 1.3 and cheek_to_jaw_ratio < 0.9:
        return 18
    elif cheek_to_jaw_ratio >= 1.0:
        return 11
    elif cheek_to_jaw_ratio < 0.8 and aspect_ratio > 1.5:
        return 15
    elif landmarks[13][0] - landmarks[3][0] > cheek_width / 2 and landmarks[8][1] > landmarks[13][1]:
        return 26
    elif cheek_width < forehead_width and aspect_ratio > 1.4:
        return 0
    else:
        return 22

def detect_face_shape(image_path, detector=detector, predictor=predictor):
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return 0
        # Resize image to reduce processing load
        height, width = image.shape[:2]
        max_dim = 800
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = detector(gray)
        if not faces:
            return 0

        for face in faces:
            # Get facial landmarks
            landmarks = predictor(gray, face)
            landmarks_points = np.array([[p.x, p.y] for p in landmarks.parts()])

            # Add virtual forehead points (above eyebrows)
            forehead_left = landmarks_points[19] - np.array([0, 30])
            forehead_right = landmarks_points[24] - np.array([0, 30])
            extended_landmarks = np.vstack([landmarks_points, forehead_left, forehead_right])

            # Determine face shape
            face_shape = calculate_face_shape(extended_landmarks)
            face_shape = face_shape*100/26
            return face_shape
        
        return 0
    except Exception as e:
        st.error(f"Error in face shape detection: {e}")
        return 0

def get_average_skin_color(image_path, detector=detector):
    try:
        # Load the image
        image = cv2.imread(image_path)
        
        if image is None:
            return 0
            
        # Resize image to reduce processing load
        height, width = image.shape[:2]
        max_dim = 800
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))
        
        # Convert the image to RGB (OpenCV uses BGR by default)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = detector(image_rgb)
        
        if len(faces) == 0:
            return 0
        
        # Sample fewer pixels for efficiency
        skin_pixels = []
        
        for face in faces:
            # Crop the region of interest (ROI) for the face
            x1, y1, x2, y2 = (face.left(), face.top(), face.right(), face.bottom())
            roi = image[y1:y2, x1:x2]
            
            # Sample pixels at regular intervals for efficiency
            h, w, _ = roi.shape
            step = max(1, min(h, w) // 20)  # Reduce sampling for better performance
            
            for i in range(0, roi.shape[0], step):
                for j in range(0, roi.shape[1], step):
                    skin_pixels.append(roi[i, j])
        
        # Convert to numpy array for easy calculation
        skin_pixels = np.array(skin_pixels)
        
        # Compute the average BGR values
        avg_bgr = np.mean(skin_pixels, axis=0)
        
        # Calculate S = sqrt(a^2 + b^2 + c^2)
        a, b, c = avg_bgr
        S = (np.sqrt(a**2 + b**2 + c**2))
        return S
    except Exception as e:
        st.error(f"Error in skin color detection: {e}")
        return 0

def rate_jawline(image_path, detector=detector, predictor=predictor):
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            return 0
            
        # Resize image to reduce processing load
        height, width = img.shape[:2]
        max_dim = 800
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            img = cv2.resize(img, (int(width * scale), int(height * scale)))
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect face
        faces = detector(gray)
        if len(faces) == 0:
            return 0
        
        # Get the landmarks of the first detected face
        face = faces[0]
        landmarks = predictor(gray, face)
        
        # Extract jawline points (points 0 to 16 in 68 landmarks)
        jawline = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(0, 17)])
        
        # Calculate sharpness and symmetry
        left_jawline = jawline[:9]  # Points 0-8
        right_jawline = jawline[8:]  # Points 8-16
        
        # Symmetry: Compare distances from left and right to the center point
        center_point = jawline[8]
        left_distances = [np.linalg.norm(pt - center_point) for pt in left_jawline]
        right_distances = [np.linalg.norm(pt - center_point) for pt in right_jawline[::-1]]  # Reverse right side

        symmetry_score = 100 - np.mean(np.abs(np.array(left_distances) - np.array(right_distances)))
        symmetry_score = max(0, min(100, symmetry_score))  # Clamp between 0 and 100
        
        # Sharpness: Angle between consecutive points
        def calculate_angle(a, b, c):
            ba = a - b
            bc = c - b
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            cosine_angle = min(1.0, max(-1.0, cosine_angle))  # Ensure value is in range for arccos
            return np.degrees(np.arccos(cosine_angle))
        
        angles = []
        for i in range(1, len(jawline) - 1):
            angles.append(calculate_angle(jawline[i - 1], jawline[i], jawline[i + 1]))
        
        sharpness_score = 100 - np.mean(np.abs(np.array(angles) - 120))  # Ideal jawline angle around 120 degrees
        sharpness_score = max(0, min(100, sharpness_score))  # Clamp between 0 and 100
        
        # Final rating (weighted average)
        jawline_rating = 0.6 * symmetry_score + 0.4 * sharpness_score
        return jawline_rating
    except Exception as e:
        st.error(f"Error in jawline detection: {e}")
        return 0

def calculate_eye_shape(eye_landmarks):
    try:
        # Calculate height and width of the eye
        width = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
        height = (np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5])) +
                np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))) / 2

        # Aspect ratio
        aspect_ratio = height / width

        # Define thresholds for different eye shapes
        if aspect_ratio < 0.25:
            return 38
        elif aspect_ratio >= 0.25 and aspect_ratio <= 0.35:
            return 28
        elif aspect_ratio > 0.35:
            return 22
        else:
            return 12
    except Exception as e:
        return 12  # Default value in case of error

def detect_eyes_shape(image_path, detector=detector, predictor=predictor):
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            return 0
            
        # Resize image to reduce processing load
        height, width = img.shape[:2]
        max_dim = 800
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            img = cv2.resize(img, (int(width * scale), int(height * scale)))
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector(gray)
        
        if not faces:
            return 0

        for face in faces:
            # Predict facial landmarks
            landmarks = predictor(gray, face)

            # Extract landmarks for both eyes
            left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

            # Determine eye shape
            left_eye_shape = calculate_eye_shape(left_eye)
            right_eye_shape = calculate_eye_shape(right_eye)

            return ((left_eye_shape+right_eye_shape)*100/72)
        
        return 0
    except Exception as e:
        st.error(f"Error in eye shape detection: {e}")
        return 0

def classify_eye_color(rgb_values):
    try:
        # Extract the R, G, B values
        r, g, b = rgb_values

        # Define color thresholds based on typical RGB values for each eye color
        if r > 100 and g < 70 and b < 40:
            return 5
        elif r > 140 and g > 100 and b < 60:
            return 19
        elif r < 100 and g < 100 and b > 120:
            return 29
        elif r < 100 and g > 120 and b < 100:
            return 14
        elif r > 100 and g > 80 and b < 60:
            return 9
        elif r < 100 and g < 100 and b < 80:
            return 24
        else:
            return 15
    except Exception as e:
        return 15  # Default value in case of error

def get_eye_rgb_value(eye_image):
    try:
        # Convert to a simple average RGB value
        avg_color = np.mean(eye_image, axis=(0, 1))  # Average of R, G, B across the eye image
        return avg_color
    except Exception as e:
        return np.array([0, 0, 0])  # Default value in case of error

def detect_eye_colors(image_path, face_cascade=face_cascade, eye_cascade=eye_cascade):
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            return 0

        # Resize image to reduce processing load
        height, width = image.shape[:2]
        max_dim = 800
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))
            
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return 0

        # Assuming the first face detected is the main face
        (x, y, w, h) = faces[0]

        # Crop the face region
        face_region = image[y:y+h, x:x+w]
        face_gray = gray[y:y+h, x:x+w]

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 3)

        # If less than two eyes are detected, return an error
        if len(eyes) < 2:
            return 0

        # Sort eyes by x-coordinate to identify left and right
        eyes = sorted(eyes, key=lambda e: e[0])
        
        # If we have more than 2 detected "eyes", just use the first two
        if len(eyes) > 2:
            eyes = eyes[:2]

        # Extract the left and right eye regions
        left_eye = face_region[eyes[0][1]:eyes[0][1]+eyes[0][3], eyes[0][0]:eyes[0][0]+eyes[0][2]]
        right_eye = face_region[eyes[1][1]:eyes[1][1]+eyes[1][3], eyes[1][0]:eyes[1][0]+eyes[1][2]]

        # Get RGB values for both eyes
        left_eye_rgb = get_eye_rgb_value(left_eye)
        right_eye_rgb = get_eye_rgb_value(right_eye)

        # Classify eye color based on RGB values
        left_eye_color = classify_eye_color(left_eye_rgb)
        right_eye_color = classify_eye_color(right_eye_rgb)

        S = (left_eye_color+right_eye_color)/2
        S = S*100/30
        return S
    except Exception as e:
        st.error(f"Error in eye color detection: {e}")
        return 0

def calculate_hair_color_score(image_path):
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            return 0, 0, 0

        # Resize image to reduce processing load
        height, width = image.shape[:2]
        max_dim = 800
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))
            
        # Convert the image to RGB (OpenCV loads images in BGR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Assume hair region is the top part of the image, focus on the top 40% for simplicity
        height, width, _ = image.shape
        hair_region = image_rgb[:int(height * 0.4), :]
        
        # Sample pixels at regular intervals for efficiency
        step = max(1, min(height, width) // 30)  # Sample fewer pixels for better performance
        
        samples = []
        for i in range(0, hair_region.shape[0], step):
            for j in range(0, hair_region.shape[1], step):
                samples.append(hair_region[i, j])
                
        # Calculate the average RGB values from samples
        avg_rgb = np.mean(samples, axis=0)
        
        a, b, c = avg_rgb[0], avg_rgb[1], avg_rgb[2]  # R, G, B components
        
        return a, b, c
    except Exception as e:
        st.error(f"Error in hair color calculation: {e}")
        return 0, 0, 0

def calculate_hair_density_and_baldness(image_path):
    try:
        # Load the image in grayscale for easier processing
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return 0, 0
        
        # Resize for efficiency if the image is large
        max_dimension = 500
        height, width = image.shape
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))
            height, width = image.shape
        
        # Use Canny edge detection to identify the edges (hair strands and scalp)
        edges = cv2.Canny(image, threshold1=100, threshold2=200)
        
        # Define a mask for the top 40% of the image (assumed hair region)
        mask = np.zeros_like(edges)
        mask[:int(height * 0.4), :] = 255  # Top 40% of the image
        
        # Calculate hair density (S1) - percentage of edge pixels in the hair region
        total_pixels = np.sum(mask > 0)
        if total_pixels == 0:
            return 0, 0
            
        hair_pixels = np.sum(cv2.bitwise_and(edges, mask) > 0)
        S1 = (hair_pixels / total_pixels) * 100  # Hair density score (S1)
        
        # Baldness score (S2) - Based on the percentage of visible scalp (i.e., lack of hair)
        scalp_pixels = np.sum(cv2.bitwise_and(edges, mask) == 0)
        S2 = (scalp_pixels / total_pixels) * 100  # Baldness score (S2)
        
        return S1, S2
    except Exception as e:
        st.error(f"Error in hair density calculation: {e}")
        return 0, 0

def calculate_final_score(image_path):
    try:
        # Step 1: Get RGB values for hair color (a, b, c)
        a, b, c = calculate_hair_color_score(image_path)
        
        # Step 2: Calculate hair density score (S1) and baldness score (S2)
        S1, S2 = calculate_hair_density_and_baldness(image_path)
        
        # Step 3: Calculate the color-based score
        if a == 0 and b == 0 and c == 0:
            return 0
            
        color_score = ((256 * 1.74 - (a**2 + b**2 + c**2)**0.5) * 100 / (256 * 1.74))
        
        # Step 4: Final score (S)
        S = (color_score + S1 + S2) / 3
        return S
    except Exception as e:
        st.error(f"Error in final hair score calculation: {e}")
        return 0

def mark_winner(image_path, is_winner=True):
    try:
        # Open the image with PIL
        image = Image.open(image_path)

        if is_winner:
            # Create a drawing context
            draw = ImageDraw.Draw(image)

            # Define the text
            text = "Hott ONE"

            # Define the font and size
            try:
                # Try loading a specific font with a reasonable size
                font = ImageFont.truetype("arial.ttf", 50)
            except IOError:
                # Fallback to default font if truetype is unavailable
                font = ImageFont.load_default()

            # Calculate the width and height of the text using textbbox
            try:
                # For newer PIL versions
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]  # width of the text
                text_height = bbox[3] - bbox[1]  # height of the text
            except AttributeError:
                # For older PIL versions
                text_width, text_height = draw.textsize(text, font=font)

            # Image dimensions
            image_width, image_height = image.size

            # Calculate the position: bottom center
            position_x = (image_width - text_width) // 2
            position_y = image_height - text_height - 20

            # Draw a white rectangle as background for the text
            rectangle_position = (position_x - 10, position_y - 5, position_x + text_width + 10, position_y + text_height + 5)
            draw.rectangle(rectangle_position, fill="white")

            # Add text
            draw.text((position_x, position_y), text, fill="black", font=font)

        return image
    except Exception as e:
        st.error(f"Error marking winner: {e}")
        return Image.open(image_path)  # Return original image on error

def analyze_image(image_path, progress_callback=None):
    try:
        # Create a dictionary to store all metrics
        metrics = {}
        
        # Run face detection first
        if progress_callback:
            progress_callback(0, "Detecting face shape...")
        metrics['face_shape'] = detect_face_shape(image_path)
        
        # If no face detected, set all metrics to zero and return early
        if metrics['face_shape'] == 0:
            metrics['skin_color'] = 0
            metrics['skin_score'] = 0
            metrics['jawline'] = 0
            metrics['eye_shape'] = 0
            metrics['eye_color'] = 0
            metrics['hair_score'] = 0
            metrics['final_score'] = 0
            
            if progress_callback:
                progress_callback(100, "No face detected!")
                
            return metrics
        
        # Continue with other metrics if face was detected
        if progress_callback:
            progress_callback(20, "Analyzing skin color...")
        metrics['skin_color'] = get_average_skin_color(image_path)
        metrics['skin_score'] = (1.5*100*metrics['skin_color']/(256*(3**0.5)))
        
        if progress_callback:
            progress_callback(35, "Analyzing jawline...")
        metrics['jawline'] = rate_jawline(image_path)
        
        if progress_callback:
            progress_callback(50, "Analyzing eye shape...")
        metrics['eye_shape'] = detect_eyes_shape(image_path)
        
        if progress_callback:
            progress_callback(65, "Analyzing eye color...")
        metrics['eye_color'] = detect_eye_colors(image_path)
        
        if progress_callback:
            progress_callback(80, "Analyzing hair...")
        metrics['hair_score'] = calculate_final_score(image_path)
        
        # Calculate composite score
        metrics['final_score'] = (
            metrics['face_shape'] * 25 + 
            metrics['skin_score'] * 40 + 
            metrics['jawline'] * 15 + 
            metrics['eye_shape'] * 10 + 
            metrics['eye_color'] * 10 + 
            metrics['hair_score'] * 20
        )
        
        if progress_callback:
            progress_callback(100, "Analysis complete!")
            
        return metrics
    except Exception as e:
        st.error(f"Error in image analysis: {e}")
        return {
            'face_shape': 0, 'skin_color': 0, 'skin_score': 0,
            'jawline': 0, 'eye_shape': 0, 'eye_color': 0,
            'hair_score': 0, 'final_score': 0
        }
# File uploader for the two images
col1, col2 = st.columns(2)

with col1:
    st.subheader("Image 1")
    uploaded_file1 = st.file_uploader("Choose first image", type=['jpg', 'jpeg', 'png'], key="file1")

with col2:
    st.subheader("Image 2")
    uploaded_file2 = st.file_uploader("Choose second image", type=['jpg', 'jpeg', 'png'], key="file2")

# Process images when both are uploaded
if uploaded_file1 is not None and uploaded_file2 is not None:
    st.write("Processing images... This may take a moment.")
    
    # Save the uploaded files to temporary locations
    temp_file1 = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_file1.write(uploaded_file1.getvalue())
    temp_file1.close()
    
    temp_file2 = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_file2.write(uploaded_file2.getvalue())
    temp_file2.close()
    
    # Progress bar and status text
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Define a callback for progress updates
        def update_progress(percent, message=""):
            progress_bar.progress(percent / 100)
            if message:
                status_text.text(message)
        
        # Process first image
        status_text.text("Analyzing first image...")
        metrics1 = analyze_image(temp_file1.name, 
            lambda percent, msg: update_progress(percent / 2, f"Image 1: {msg}")
        )
        
        # Process second image
        status_text.text("Analyzing second image...")
        metrics2 = analyze_image(temp_file2.name, 
            lambda percent, msg: update_progress(50 + percent / 2, f"Image 2: {msg}")
        )
        
        status_text.text("Comparing results...")
        
        # Extract scores
        s1 = metrics1['final_score']
        s2 = metrics2['final_score']
        
        # Determine winner
        winner_image = temp_file1.name if s1 >= s2 else temp_file2.name
        loser_image = temp_file2.name if s1 >= s2 else temp_file1.name
        
        # Create result images
        winner_pil = mark_winner(winner_image, True)
        
        # Display results
        status_text.empty()
        st.subheader("Results")
        col3, col4 = st.columns(2)
        
        with col3:
            st.image(uploaded_file1, caption="Image 1")
            st.metric("Score", f"{s1:.2f}")
            
            # Show detailed breakdown for Image 1
            with st.expander("Detailed Scores for Image 1"):
                st.write(f"Face Shape: {metrics1['face_shape']:.2f}")
                st.write(f"Skin Score: {metrics1['skin_score']:.2f}")
                st.write(f"Jawline: {metrics1['jawline']:.2f}")
                st.write(f"Eye Shape: {metrics1['eye_shape']:.2f}")
                st.write(f"Eye Color: {metrics1['eye_color']:.2f}")
                st.write(f"Hair Score: {metrics1['hair_score']:.2f}")
        
        with col4:
            st.image(uploaded_file2, caption="Image 2") 
            st.metric("Score", f"{s2:.2f}")
            
            # Show detailed breakdown for Image 2
            with st.expander("Detailed Scores for Image 2"):
                st.write(f"Face Shape: {metrics2['face_shape']:.2f}")
                st.write(f"Skin Score: {metrics2['skin_score']:.2f}")
                st.write(f"Jawline: {metrics2['jawline']:.2f}")
                st.write(f"Eye Shape: {metrics2['eye_shape']:.2f}")
                st.write(f"Eye Color: {metrics2['eye_color']:.2f}")
                st.write(f"Hair Score: {metrics2['hair_score']:.2f}")
        
        # Display the winner
        st.subheader("Winner")
        if s1 >= s2:
            st.image(winner_pil, caption="Image 1 Wins!")
        else:
            st.image(winner_pil, caption="Image 2 Wins!")
            
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
    
    finally:
        # Clean up the temporary files
        try:
            os.unlink(temp_file1.name)
            os.unlink(temp_file2.name)
        except:
            pass  # Ignore errors during cleanup

else:
    st.info("Please upload both images to compare.")
