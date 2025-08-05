import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
import threading
import time
from collections import deque, Counter
import os
import tempfile
import base64
from io import BytesIO
from datetime import datetime

# Import your main classifier (assuming it's in the same directory)
try:
    from model import UltimateTrashClassifier  # Import your main class
except ImportError:
    # If import fails, we'll define a simplified version here
    st.error("Could not import UltimateTrashClassifier. Please ensure model.py is in the same directory.")

class StreamlitTrashInterface:
    def __init__(self):
        """Initialize the Streamlit interface"""
        self.classifier = None
        self.camera_active = False
        self.current_frame = None
        self.latest_results = None
        
        # Initialize session state
        if 'detection_mode' not in st.session_state:
            st.session_state.detection_mode = 'full_frame'
        if 'confidence_threshold' not in st.session_state:
            st.session_state.confidence_threshold = 0.35
        if 'show_all_predictions' not in st.session_state:
            st.session_state.show_all_predictions = False
        if 'show_confidence_bars' not in st.session_state:
            st.session_state.show_confidence_bars = True
        if 'show_detection_regions' not in st.session_state:
            st.session_state.show_detection_regions = True
        if 'camera_running' not in st.session_state:
            st.session_state.camera_running = False
        if 'captured_images' not in st.session_state:
            st.session_state.captured_images = []
        if 'camera_index' not in st.session_state:
            st.session_state.camera_index = 0
        if 'last_error' not in st.session_state:
            st.session_state.last_error = None
        
    def initialize_classifier(self):
        """Initialize the trash classifier"""
        if self.classifier is None:
            try:
                self.classifier = UltimateTrashClassifier()
                if hasattr(self.classifier, 'model') and self.classifier.model is not None:
                    # Update color scheme for better visibility
                    self.classifier.colors.update({
                        'battery': (0, 255, 255),      # Yellow
                        'biological': (0, 255, 0),     # Green  
                        'brown-glass': (19, 69, 139),  # Brown
                        'cardboard': (139, 69, 19),    # Dark orange
                        'green glass': (0, 128, 0),    # Dark green
                        'metal': (128, 128, 128),      # Gray
                        'paper': (255, 255, 255),      # White
                        'plastic': (0, 0, 255),        # Red
                        'shoes': (128, 0, 128),        # Purple
                        'masks': (255, 128, 0),        # Orange
                        'white glass': (200, 200, 200), # Light gray
                        'clothes': (255, 0, 255),      # Magenta
                    })
                    return True
                else:
                    st.session_state.last_error = "Failed to load the classification model"
                    return False
            except Exception as e:
                st.session_state.last_error = f"Error initializing classifier: {e}"
                return False
        return True
    
    def get_available_cameras(self):
        """Get list of available cameras"""
        available_cameras = []
        for i in range(4):  # Check cameras 0-3
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        return available_cameras
    
    def create_sidebar_controls(self):
        """Create the sidebar with all controls"""
        st.sidebar.title("🗑️ Trash Classifier Controls")
        
        # Error display
        if st.session_state.last_error:
            st.sidebar.error(st.session_state.last_error)
            if st.sidebar.button("Clear Error"):
                st.session_state.last_error = None
                st.experimental_rerun()
        
        # Camera Selection
        st.sidebar.subheader("📹 Camera Selection")
        available_cameras = self.get_available_cameras()
        
        if available_cameras:
            camera_options = {f"Camera {i}": i for i in available_cameras}
            selected_camera_name = st.sidebar.selectbox(
                "Choose camera:",
                options=list(camera_options.keys()),
                index=0
            )
            new_camera_index = camera_options[selected_camera_name]
            
            # If camera changed, stop current stream
            if new_camera_index != st.session_state.camera_index:
                if st.session_state.camera_running:
                    st.session_state.camera_running = False
                st.session_state.camera_index = new_camera_index
        else:
            st.sidebar.warning("No cameras detected")
            st.sidebar.info("Make sure your camera is connected and not being used by another application")
        
        st.sidebar.divider()
        
        # Detection Mode Selection
        st.sidebar.subheader("🎯 Detection Mode")
        mode_options = {
            'Full Frame (Fastest)': 'full_frame',
            'Grid Detection': 'grid', 
            'Sliding Window': 'sliding_window',
            'Hybrid Mode (Slowest)': 'hybrid'
        }
        
        selected_mode = st.sidebar.selectbox(
            "Choose detection mode:",
            options=list(mode_options.keys()),
            index=0,
            help="Full Frame: Classify entire view | Grid: Divide into sections | Sliding Window: Scan with windows | Hybrid: Combine all methods"
        )
        
        new_mode = mode_options[selected_mode]
        if new_mode != st.session_state.detection_mode:
            st.session_state.detection_mode = new_mode
        
        # Update classifier detection modes
        if self.classifier:
            self.classifier.detection_modes = {k: False for k in self.classifier.detection_modes}
            if st.session_state.detection_mode == 'hybrid':
                self.classifier.detection_modes['grid'] = True
                self.classifier.detection_modes['sliding_window'] = True
                self.classifier.detection_modes['hybrid'] = True
            else:
                self.classifier.detection_modes[st.session_state.detection_mode] = True
        
        st.sidebar.divider()
        
        # Display Options
        st.sidebar.subheader("📊 Display Options")
        new_show_all = st.sidebar.checkbox(
            "Show All Predictions", 
            value=st.session_state.show_all_predictions,
            help="Display confidence scores for all classes"
        )
        new_show_bars = st.sidebar.checkbox(
            "Show Confidence Bars", 
            value=st.session_state.show_confidence_bars,
            help="Show visual confidence bars"
        )
        new_show_regions = st.sidebar.checkbox(
            "Show Detection Regions", 
            value=st.session_state.show_detection_regions,
            help="Highlight detected object regions"
        )
        
        # Update session state
        st.session_state.show_all_predictions = new_show_all
        st.session_state.show_confidence_bars = new_show_bars
        st.session_state.show_detection_regions = new_show_regions
        
        # Update classifier display options
        if self.classifier:
            self.classifier.show_all_predictions = st.session_state.show_all_predictions
            self.classifier.show_confidence_bars = st.session_state.show_confidence_bars
            self.classifier.show_detection_regions = st.session_state.show_detection_regions
        
        st.sidebar.divider()
        
        # Confidence Threshold
        st.sidebar.subheader("🎚️ Confidence Threshold")
        new_threshold = st.sidebar.slider(
            "Minimum confidence for detections:",
            min_value=0.05,
            max_value=0.95,
            value=st.session_state.confidence_threshold,
            step=0.05,
            format="%.2f",
            help="Lower values detect more objects but may include false positives"
        )
        
        if new_threshold != st.session_state.confidence_threshold:
            st.session_state.confidence_threshold = new_threshold
        
        # Update classifier threshold
        if self.classifier:
            self.classifier.confidence_threshold = st.session_state.confidence_threshold
        
        st.sidebar.divider()
        
        # Camera Controls
        st.sidebar.subheader("📹 Camera Controls")
        
        if not available_cameras:
            st.sidebar.error("No cameras available")
        else:
            col1, col2 = st.sidebar.columns(2)
            
            with col1:
                if st.button("▶️ Start", disabled=st.session_state.camera_running, key="start_camera"):
                    st.session_state.camera_running = True
                    st.session_state.last_error = None
                    st.experimental_rerun()
            
            with col2:
                if st.button("⏹️ Stop", disabled=not st.session_state.camera_running, key="stop_camera"):
                    st.session_state.camera_running = False
                    st.experimental_rerun()
        
        # Capture Controls
        if st.sidebar.button("📸 Capture Frame", disabled=not st.session_state.camera_running, key="capture_frame"):
            self.capture_current_frame()
        
        if st.sidebar.button("🗑️ Clear Captures", disabled=len(st.session_state.captured_images)==0, key="clear_captures"):
            st.session_state.captured_images = []
            st.experimental_rerun()
        
        # File Upload Option
        st.sidebar.divider()
        st.sidebar.subheader("📁 Upload Image")
        uploaded_file = st.sidebar.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Upload an image to classify (PNG, JPG, JPEG, BMP)"
        )
        
        if uploaded_file is not None:
            self.process_uploaded_image(uploaded_file)
    
    def process_uploaded_image(self, uploaded_file):
        """Process an uploaded image file"""
        if not self.initialize_classifier():
            st.error(st.session_state.last_error)
            return
        
        try:
            # Read the uploaded image
            image = Image.open(uploaded_file)
            
            # Convert PIL image to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Process the image
            results = self.process_single_frame(opencv_image)
            
            # Draw results on the image
            if results:
                opencv_image = self.draw_results_on_frame(opencv_image, results)
            
            # Display results
            st.subheader("📋 Uploaded Image Results")
            
            # Convert back to RGB for display
            display_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(display_image, caption=f"Results for {uploaded_file.name}", use_column_width=True)
            
            with col2:
                if results:
                    self.display_classification_results(results)
                    
        except Exception as e:
            st.error(f"Error processing uploaded image: {e}")
    
    def capture_current_frame(self):
        """Capture the current frame"""
        if self.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Convert frame to RGB for display
            rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            
            # Save to session state
            capture_data = {
                'timestamp': timestamp,
                'image': rgb_frame,
                'results': self.latest_results
            }
            st.session_state.captured_images.append(capture_data)
            
            st.success(f"📸 Frame captured at {timestamp}")
            st.experimental_rerun()
    
    def process_single_frame(self, frame):
        """Process a single frame and return results"""
        if not self.classifier:
            return None
        
        try:
            results = {}
            
            # Process based on current mode
            if st.session_state.detection_mode == 'full_frame':
                class_name, confidence, all_predictions = self.classifier.full_frame_classification(frame)
                results = {
                    'type': 'full_frame',
                    'class_name': class_name,
                    'confidence': confidence,
                    'all_predictions': all_predictions
                }
            else:
                # Multi-object detection
                detections = []
                if st.session_state.detection_mode == 'grid':
                    detections = self.classifier.grid_detection(frame)
                elif st.session_state.detection_mode == 'sliding_window':
                    detections = self.classifier.sliding_window_detection(frame)
                elif st.session_state.detection_mode == 'hybrid':
                    detections = self.classifier.hybrid_detection(frame)
                
                if detections:
                    detections = self.classifier.non_max_suppression(detections)
                
                results = {
                    'type': 'multi_object',
                    'detections': detections,
                    'object_counts': Counter([det['class'] for det in detections])
                }
            
            return results
            
        except Exception as e:
            st.session_state.last_error = f"Error processing frame: {e}"
            return None
    
    def display_classification_results(self, results):
        """Display classification results in the UI"""
        if not results:
            return
        
        if results['type'] == 'full_frame':
            # Full frame results
            class_name = results['class_name']
            confidence = results['confidence']
            
            # Create a nice metric display
            st.metric(
                label="🎯 Primary Classification",
                value=class_name.upper(),
                delta=f"{confidence:.1%} confidence"
            )
            
            # Confidence progress bar
            if st.session_state.show_confidence_bars:
                st.progress(confidence, text=f"Confidence: {confidence:.1%}")
            
            # All predictions
            if st.session_state.show_all_predictions and results['all_predictions'] is not None:
                st.subheader("📊 All Class Predictions")
                
                # Create prediction dataframe
                pred_data = []
                for i, (class_name_raw, pred_conf) in enumerate(zip(self.classifier.class_names, results['all_predictions'])):
                    clean_name = self.classifier.clean_class_name(class_name_raw)
                    if pred_conf > 0.03:  # Show predictions above 3%
                        pred_data.append({
                            'Class': clean_name.title(),
                            'Confidence': f"{pred_conf:.1%}",
                            'Score': pred_conf
                        })
                
                # Sort by confidence
                pred_data.sort(key=lambda x: x['Score'], reverse=True)
                
                # Display top predictions in a more compact way
                for i, pred in enumerate(pred_data[:8]):  # Show top 8
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"**{i+1}. {pred['Class']}**")
                    with col2:
                        st.write(pred['Confidence'])
                    with col3:
                        if st.session_state.show_confidence_bars:
                            st.progress(pred['Score'], text="")
        
        elif results['type'] == 'multi_object':
            # Multi-object detection results
            detections = results['detections']
            object_counts = results['object_counts']
            
            st.metric(
                label="🔍 Objects Detected",
                value=len(detections),
                delta=f"{len(object_counts)} different types" if len(object_counts) > 1 else ""
            )
            
            if object_counts:
                st.subheader("📈 Object Summary")
                
                # Create a nice summary table
                for class_name, count in sorted(object_counts.items()):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{class_name.title()}**")
                    with col2:
                        st.write(f"**{count}** item{'s' if count > 1 else ''}")
            
            # Detailed detections in expandable format
            if detections:
                st.subheader("🔍 Detection Details")
                for i, detection in enumerate(detections):
                    confidence_emoji = "🟢" if detection['confidence'] > 0.7 else "🟡" if detection['confidence'] > 0.5 else "🟠"
                    with st.expander(f"{confidence_emoji} Detection {i+1}: {detection['class'].title()} ({detection['confidence']:.1%})"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Class:** {detection['class'].title()}")
                            st.write(f"**Confidence:** {detection['confidence']:.1%}")
                        with col2:
                            st.write(f"**Method:** {detection.get('method', 'unknown').title()}")
                            bbox = detection['bbox']
                            st.write(f"**Position:** ({bbox[0]}, {bbox[1]}) → ({bbox[2]}, {bbox[3]})")
    
    def run_camera_stream(self):
        """Run the camera stream"""
        if not st.session_state.camera_running:
            return
        
        if not self.initialize_classifier():
            st.error(st.session_state.last_error)
            st.session_state.camera_running = False
            return
        
        # Camera stream placeholder
        camera_placeholder = st.empty()
        results_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Initialize camera
        cap = cv2.VideoCapture(st.session_state.camera_index)
        
        if not cap.isOpened():
            st.session_state.last_error = f"❌ Could not open Camera {st.session_state.camera_index}"
            st.session_state.camera_running = False
            st.experimental_rerun()
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        fps_counter = deque(maxlen=30)
        last_time = time.time()
        
        try:
            while st.session_state.camera_running:
                ret, frame = cap.read()
                if not ret:
                    st.session_state.last_error = "❌ Could not read frame from camera"
                    break
                
                # Calculate FPS
                current_time = time.time()
                frame_time = current_time - last_time
                fps_counter.append(1.0 / frame_time if frame_time > 0 else 0)
                last_time = current_time
                
                self.current_frame = frame.copy()
                
                # Process frame every few frames for performance
                process_interval = 2 if st.session_state.detection_mode == 'full_frame' else 5
                if frame_count % process_interval == 0:
                    self.latest_results = self.process_single_frame(frame)
                
                # Draw results on frame
                if self.latest_results:
                    frame = self.draw_results_on_frame(frame, self.latest_results)
                
                # Draw FPS counter
                if fps_counter:
                    avg_fps = np.mean(fps_counter)
                    cv2.putText(frame, f"FPS: {avg_fps:.1f}", (frame.shape[1] - 100, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Convert frame to RGB for display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Display frame
                camera_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
                
                # Display results
                if self.latest_results:
                    with results_placeholder.container():
                        self.display_classification_results(self.latest_results)
                
                # Status info
                with status_placeholder.container():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Camera", f"#{st.session_state.camera_index}")
                    with col2:
                        st.metric("FPS", f"{np.mean(fps_counter):.1f}" if fps_counter else "0.0")
                    with col3:
                        st.metric("Frame", frame_count)
                
                frame_count += 1
                time.sleep(0.033)  # ~30 FPS
                
        except Exception as e:
            st.session_state.last_error = f"Camera error: {e}"
        
        finally:
            cap.release()
            st.session_state.camera_running = False
    
    def draw_results_on_frame(self, frame, results):
        """Draw results on the frame with smaller fonts"""
        if results['type'] == 'full_frame':
            # Draw main classification with smaller font
            class_name = results['class_name']
            confidence = results['confidence']
            
            # Get color for class
            color = self.classifier.colors.get(class_name.lower(), (255, 255, 255))
            
            # Draw classification box (smaller)
            cv2.rectangle(frame, (10, 10), (300, 50), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (300, 50), color, 2)
            cv2.putText(frame, f"{class_name.upper()}: {confidence:.1%}", 
                       (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)  # Smaller font
        
        elif results['type'] == 'multi_object':
            # Draw detections with smaller fonts
            detections = results['detections']
            for i, detection in enumerate(detections):
                class_name = detection['class']
                confidence = detection['confidence']
                x1, y1, x2, y2 = detection['bbox']
                
                color = self.classifier.colors.get(class_name.lower(), (255, 255, 255))
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with smaller font
                label = f"{class_name}: {confidence:.1%}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]  # Smaller font
                
                # Background for label
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 8), 
                             (x1 + label_size[0] + 4, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)  # Smaller font
                
                # Draw detection number (smaller)
                cv2.circle(frame, detection['center'], 12, color, -1)
                cv2.putText(frame, str(i + 1), 
                           (detection['center'][0] - 4, detection['center'][1] + 4), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)  # Smaller font
        
        return frame
    
    def display_captured_images(self):
        """Display captured images"""
        if st.session_state.captured_images:
            st.subheader(f"📸 Captured Images ({len(st.session_state.captured_images)})")
            
            # Display in a grid layout
            cols = st.columns(2)
            for i, capture in enumerate(st.session_state.captured_images):
                with cols[i % 2]:
                    with st.expander(f"Capture {i+1} - {capture['timestamp']}", expanded=i<2):
                        st.image(capture['image'], caption=f"Captured at {capture['timestamp']}")
                        
                        if capture['results']:
                            self.display_classification_results(capture['results'])
                        
                        # Download button
                        img_pil = Image.fromarray(capture['image'])
                        buf = BytesIO()
                        img_pil.save(buf, format='PNG')
                        img_bytes = buf.getvalue()
                        
                        st.download_button(
                            label="💾 Download",
                            data=img_bytes,
                            file_name=f"trash_capture_{capture['timestamp']}.png",
                            mime="image/png",
                            key=f"download_{i}"
                        )
    
    def run_interface(self):
        """Run the main Streamlit interface"""
        # Page configuration
        st.set_page_config(
            page_title="Ultimate Trash Classifier",
            page_icon="🗑️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Main title
        st.title("🗑️ Ultimate Trash Classification System")
        st.markdown("**Real-time trash detection and classification using computer vision**")
        
        # Create sidebar controls
        self.create_sidebar_controls()
        
        # Main content area
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Status indicators
            status_col1, status_col2, status_col3, status_col4 = st.columns(4)
            
            with status_col1:
                model_status = "✅ Ready" if self.initialize_classifier() else "❌ Error"
                st.metric("Model", model_status)
            
            with status_col2:
                camera_status = "📹 Active" if st.session_state.camera_running else "📷 Idle"
                st.metric("Camera", camera_status)
            
            with status_col3:
                mode_display = st.session_state.detection_mode.replace('_', ' ').title()
                st.metric("Mode", mode_display)
            
            with status_col4:
                st.metric("Threshold", f"{st.session_state.confidence_threshold:.2f}")
            
            # Camera stream or instructions
            if st.session_state.camera_running:
                st.subheader(f"📹 Live Feed - Camera {st.session_state.camera_index}")
                self.run_camera_stream()
            else:
                st.info("👈 Use the sidebar controls to start the camera or upload an image")
                
                # Show example of what the system can detect
                st.subheader("🏷️ Detectable Trash Categories")
                
                if self.classifier and hasattr(self.classifier, 'class_names'):
                    categories = [self.classifier.clean_class_name(name) for name in self.classifier.class_names]
                    
                    # Display categories in a nice grid with emojis
                    category_emojis = {
                        'battery': '🔋', 'biological': '🍃', 'brown-glass': '🟤',
                        'cardboard': '📦', 'green glass': '💚', 'metal': '🔩',
                        'paper': '📄', 'plastic': '🥤', 'shoes': '👟',
                        'masks': '😷', 'white glass': '🤍', 'clothes': '👕'
                    }
                    
                    cols = st.columns(4)
                    for i, category in enumerate(categories):
                        with cols[i % 4]:
                            emoji = category_emojis.get(category.lower(), '🗑️')
                            st.write(f"{emoji} {category.title()}")
        
        with col2:
            st.subheader("ℹ️ System Info")
            
            # Current settings in a nice format
            st.write("**Current Configuration:**")
            st.write(f"🎯 **Mode:** {st.session_state.detection_mode.replace('_', ' ').title()}")
            st.write(f"📹 **Camera:** #{st.session_state.camera_index}")
            st.write(f"🎚️ **Threshold:** {st.session_state.confidence_threshold:.2f}")
            st.write(f"📊 **All Predictions:** {'✅' if st.session_state.show_all_predictions else '❌'}")
            st.write(f"📈 **Confidence Bars:** {'✅' if st.session_state.show_confidence_bars else '❌'}")
            
            # Performance tips
            st.subheader("💡 Tips")
            st.write("🟢 **Full Frame:** Fastest for single objects")
            st.write("🟡 **Grid:** Good for organized layouts")  
            st.write("🟠 **Sliding Window:** Best for small/multiple objects")
            st.write("🔴 **Hybrid:** Most accurate but slowest")
            
            st.subheader("🎛️ Quick Actions")
            if st.button("🔄 Reset All Settings", key="reset_all"):
                st.session_state.detection_mode = 'full_frame'
                st.session_state.confidence_threshold = 0.35
                st.session_state.show_all_predictions = False
                st.session_state.show_confidence_bars = True
                st.session_state.show_detection_regions = True
                st.experimental_rerun()
        
        # Display captured images
        self.display_captured_images()
        
        # Footer
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Built with:** Streamlit + OpenCV + TensorFlow")
        with col2:
            if self.classifier and hasattr(self.classifier, 'class_names'):
                st.markdown(f"**Classes:** {len(self.classifier.class_names)}")
            else:
                st.markdown("**Classes:** 12")
        with col3:
            st.markdown("**Version:** Ultimate Trash Classifier v2.0")

def main():
    """Main function to run the Streamlit app"""
    interface = StreamlitTrashInterface()
    interface.run_interface()

if __name__ == "__main__":
    main()