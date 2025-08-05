import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
import threading
import time
from collections import deque, Counter
import os
import math
import json


class UltimateTrashClassifier:
    def __init__(self, model_path="keras_Model.h5", labels_path="labels.txt"):
        """Initialize the ultimate trash classifier combining all features"""
        np.set_printoptions(suppress=True)
        
        # Load the model
        try:
            self.model = load_model(model_path, compile=False)
            print("✅ Keras model loaded successfully")
            print("📝 Note: This uses your custom Keras .h5 model, not CLIP")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return
        
        # Load the labels
        try:
            with open(labels_path, "r") as f:
                self.class_names = [line.strip() for line in f.readlines()]
            print(f"✅ Labels loaded: {len(self.class_names)} classes")
            print(f"📋 Classes: {[name.split(' ', 1)[-1] if ' ' in name else name for name in self.class_names]}")
        except Exception as e:
            print(f"❌ Error loading labels: {e}")
            return
        
        # Create the array for model input
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        
        # Detection modes
        self.detection_modes = {
            'full_frame': True,      # Classify entire frame
            'grid': False,           # Grid-based detection
            'sliding_window': False, # Sliding window detection
            'hybrid': False          # Combination approach
        }
        
        # Parameters
        self.grid_size = 3
        self.overlap_ratio = 0.5
        self.confidence_threshold = 0.35
        self.window_sizes = [(120, 120), (180, 180), (240, 240)]
        
        # History for smoothing
        self.full_frame_history = deque(maxlen=8)
        self.detection_history = deque(maxlen=5)
        
        # Colors for visualization
        self.colors = {
            'battery': (0, 255, 255),      # Yellow
            'biological': (0, 255, 0),     # Green  
            'brown-glass': (19, 69, 139),  # Brown
            'cardboard': (139, 69, 19),    # Dark orange
            'clothes': (255, 0, 255),      # Magenta
            'green-glass': (0, 128, 0),    # Dark green
            'metal': (128, 128, 128),      # Gray
            'paper': (255, 255, 255),      # White
            'plastic': (0, 0, 255),        # Red
            'shoes': (128, 0, 128),        # Purple
            'trash': (64, 64, 64),         # Dark gray
            'white-glass': (200, 200, 200) # Light gray
        }
        
        # Performance tracking
        self.frame_times = deque(maxlen=30)
        self.prediction_times = deque(maxlen=30)
        
        # UI state
        self.show_all_predictions = False
        self.show_confidence_bars = True
        self.show_detection_regions = True
        
    def clean_class_name(self, class_name):
        """Clean class name by removing index prefix"""
        if len(class_name) > 2 and class_name[1] == ' ':
            return class_name[2:].strip()
        return class_name.strip()
    
    def preprocess_image(self, image_region):
        """Preprocess image region for model prediction"""
        try:
            # Handle different input formats
            if len(image_region.shape) == 3:
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(image_region, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image_region
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_image)
            
            # Resize and crop from center
            size = (224, 224)
            pil_image = ImageOps.fit(pil_image, size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            image_array = np.asarray(pil_image)
            
            # Normalize the image (same as your original code)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            
            return normalized_image_array
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def predict_single_region(self, image_region):
        """Predict trash type for a single image region"""
        start_time = time.time()
        
        processed_image = self.preprocess_image(image_region)
        if processed_image is None:
            return "unknown", 0.0, None
        
        try:
            # Load the image into the data array
            self.data[0] = processed_image
            
            # Make prediction
            prediction = self.model.predict(self.data, verbose=0)
            index = np.argmax(prediction)
            confidence_score = prediction[0][index]
            
            # Get class name
            class_name = self.clean_class_name(self.class_names[index])
            
            # Track prediction time
            self.prediction_times.append(time.time() - start_time)
            
            return class_name, confidence_score, prediction[0]
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return "unknown", 0.0, None
    
    def full_frame_classification(self, frame):
        """Classify the entire frame"""
        class_name, confidence, all_predictions = self.predict_single_region(frame)
        
        # Add to history for smoothing
        if all_predictions is not None:
            self.full_frame_history.append(all_predictions)
            
            # Get smoothed predictions
            if len(self.full_frame_history) > 3:
                smoothed_predictions = np.mean(self.full_frame_history, axis=0)
                smoothed_index = np.argmax(smoothed_predictions)
                smoothed_class = self.clean_class_name(self.class_names[smoothed_index])
                smoothed_confidence = smoothed_predictions[smoothed_index]
                return smoothed_class, smoothed_confidence, smoothed_predictions
        
        return class_name, confidence, all_predictions
    
    def grid_detection(self, frame):
        """Detect objects using grid-based approach"""
        height, width = frame.shape[:2]
        detections = []
        
        cell_height = height // self.grid_size
        cell_width = width // self.grid_size
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                y1 = i * cell_height
                y2 = min((i + 1) * cell_height, height)
                x1 = j * cell_width
                x2 = min((j + 1) * cell_width, width)
                
                # Extract cell region
                cell = frame[y1:y2, x1:x2]
                
                if cell.shape[0] < 50 or cell.shape[1] < 50:
                    continue
                
                class_name, confidence, _ = self.predict_single_region(cell)
                
                if confidence > self.confidence_threshold and class_name != "unknown":
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': (x1, y1, x2, y2),
                        'method': 'grid',
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                    })
        
        return detections
    
    def sliding_window_detection(self, frame):
        """Detect objects using sliding window approach"""
        height, width = frame.shape[:2]
        detections = []
        
        for window_size in self.window_sizes:
            w_width, w_height = window_size
            step_x = int(w_width * (1 - self.overlap_ratio))
            step_y = int(w_height * (1 - self.overlap_ratio))
            
            for y in range(0, height - w_height + 1, step_y):
                for x in range(0, width - w_width + 1, step_x):
                    region = frame[y:y + w_height, x:x + w_width]
                    
                    if region.shape[0] < 80 or region.shape[1] < 80:
                        continue
                    
                    class_name, confidence, _ = self.predict_single_region(region)
                    
                    if confidence > self.confidence_threshold and class_name != "unknown":
                        detections.append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': (x, y, x + w_width, y + w_height),
                            'method': 'sliding',
                            'window_size': window_size,
                            'center': (x + w_width // 2, y + w_height // 2)
                        })
        
        return detections
    
    def hybrid_detection(self, frame):
        """Combine multiple detection methods"""
        all_detections = []
        
        # Get detections from both methods
        if self.detection_modes['grid']:
            all_detections.extend(self.grid_detection(frame))
        
        if self.detection_modes['sliding_window']:
            all_detections.extend(self.sliding_window_detection(frame))
        
        # Apply non-maximum suppression
        return self.non_max_suppression(all_detections)
    
    def non_max_suppression(self, detections, overlap_threshold=0.4):
        """Apply non-maximum suppression"""
        if len(detections) == 0:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            current = detections.pop(0)
            keep.append(current)
            
            remaining = []
            for det in detections:
                if self.calculate_iou(current['bbox'], det['bbox']) < overlap_threshold:
                    remaining.append(det)
            detections = remaining
        
        return keep
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def draw_full_frame_results(self, frame, class_name, confidence, all_predictions=None):
        """Draw full frame classification results"""
        height, width = frame.shape[:2]
        
        # Main prediction box
        main_color = self.colors.get(class_name.lower(), (255, 255, 255))
        
        # Draw main result
        cv2.rectangle(frame, (10, 10), (450, 90), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (450, 90), main_color, 3)
        
        text = f"PRIMARY: {class_name.upper()}"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, main_color, 2)
        
        confidence_text = f"CONFIDENCE: {confidence:.1%}"
        cv2.putText(frame, confidence_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, main_color, 2)
        
        # Show confidence bar
        if self.show_confidence_bars:
            bar_width = int(400 * confidence)
            cv2.rectangle(frame, (20, 75), (20 + bar_width, 85), main_color, -1)
            cv2.rectangle(frame, (20, 75), (420, 85), main_color, 1)
        
        # Show all predictions if enabled
        if self.show_all_predictions and all_predictions is not None:
            y_offset = 110
            cv2.putText(frame, "ALL PREDICTIONS:", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            
            # Sort predictions by confidence
            pred_pairs = list(zip(self.class_names, all_predictions))
            pred_pairs.sort(key=lambda x: x[1], reverse=True)
            
            for i, (pred_class, pred_conf) in enumerate(pred_pairs[:6]):  # Show top 6
                pred_class = self.clean_class_name(pred_class)
                if pred_conf > 0.05:  # Only show predictions above 5%
                    color = self.colors.get(pred_class.lower(), (255, 255, 255))
                    text = f"{pred_class}: {pred_conf:.1%}"
                    cv2.putText(frame, text, (20, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                    # Mini confidence bar
                    if self.show_confidence_bars:
                        bar_width = int(100 * pred_conf)
                        cv2.rectangle(frame, (200, y_offset - 8), (200 + bar_width, y_offset - 3), color, -1)
                    
                    y_offset += 18
    
    def draw_detections(self, frame, detections):
        """Draw detection results"""
        height, width = frame.shape[:2]
        
        # Count objects by type
        object_counts = Counter([det['class'] for det in detections])
        
        # Draw bounding boxes and labels
        for i, detection in enumerate(detections):
            class_name = detection['class']
            confidence = detection['confidence']
            x1, y1, x2, y2 = detection['bbox']
            method = detection.get('method', 'unknown')
            
            color = self.colors.get(class_name.lower(), (255, 255, 255))
            
            # Draw bounding box with different styles for different methods
            if method == 'grid':
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            else:  # sliding window
                # Dashed line effect
                for j in range(x1, x2, 10):
                    cv2.line(frame, (j, y1), (min(j + 5, x2), y1), color, 2)
                    cv2.line(frame, (j, y2), (min(j + 5, x2), y2), color, 2)
                for j in range(y1, y2, 10):
                    cv2.line(frame, (x1, j), (x1, min(j + 5, y2)), color, 2)
                    cv2.line(frame, (x2, j), (x2, min(j + 5, y2)), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.1%}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Ensure label fits in frame
            label_y = max(y1, label_size[1] + 5)
            cv2.rectangle(frame, (x1, label_y - label_size[1] - 5), 
                         (x1 + label_size[0] + 5, label_y), color, -1)
            cv2.putText(frame, label, (x1 + 2, label_y - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Draw detection number
            cv2.circle(frame, detection['center'], 15, color, -1)
            cv2.putText(frame, str(i + 1), (detection['center'][0] - 5, detection['center'][1] + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Draw summary panel
        if detections:
            panel_height = len(object_counts) * 22 + 80
            panel_width = 280
            cv2.rectangle(frame, (width - panel_width - 10, height - panel_height - 10), 
                         (width - 10, height - 10), (0, 0, 0), -1)
            cv2.rectangle(frame, (width - panel_width - 10, height - panel_height - 10), 
                         (width - 10, height - 10), (255, 255, 255), 2)
            
            y_offset = height - panel_height + 10
            cv2.putText(frame, f"DETECTED OBJECTS: {len(detections)}", 
                       (width - panel_width, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            
            for class_name, count in sorted(object_counts.items()):
                color = self.colors.get(class_name.lower(), (255, 255, 255))
                text = f"{class_name}: {count}"
                cv2.putText(frame, text, (width - panel_width, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                y_offset += 20
    
    def draw_ui_info(self, frame):
        """Draw UI information and controls"""
        height, width = frame.shape[:2]
        
        # Current mode indicator
        active_modes = [mode for mode, active in self.detection_modes.items() if active]
        mode_text = f"MODE: {', '.join(active_modes).upper()}"
        cv2.putText(frame, mode_text, (10, height - 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Performance info
        if self.frame_times:
            fps = 1.0 / np.mean(self.frame_times)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, height - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if self.prediction_times:
            avg_pred_time = np.mean(self.prediction_times) * 1000
            cv2.putText(frame, f"Pred: {avg_pred_time:.1f}ms", (10, height - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Confidence threshold
        cv2.putText(frame, f"Threshold: {self.confidence_threshold:.2f}", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Controls
        controls = [
            "Q: Quit | 1: Full Frame | 2: Grid | 3: Sliding | 4: Hybrid",
            "A: All Predictions | B: Confidence Bars | R: Detection Regions",
            "+/-: Threshold | C: Capture | SPACE: Pause"
        ]
        
        for i, control in enumerate(controls):
            cv2.putText(frame, control, (10, 30 + i * 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    
    def run_ultimate_classifier(self, camera_index=0):
        """Run the ultimate trash classification system"""
        # Initialize camera
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("🚀 Ultimate Trash Classifier Started!")
        print("\n📋 CONTROLS:")
        print("  1 - Full Frame Mode (classify entire view)")
        print("  2 - Grid Detection Mode")
        print("  3 - Sliding Window Mode")
        print("  4 - Hybrid Mode (grid + sliding)")
        print("  A - Toggle all predictions display")
        print("  B - Toggle confidence bars")
        print("  R - Toggle detection regions")
        print("  + - Increase confidence threshold")
        print("  - - Decrease confidence threshold")
        print("  C - Capture current frame")
        print("  SPACE - Pause/Resume")
        print("  Q - Quit")
        
        frame_count = 0
        paused = False
        last_frame_time = time.time()
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame")
                    break
                
                # Track frame time
                current_time = time.time()
                self.frame_times.append(current_time - last_frame_time)
                last_frame_time = current_time
                
                # Process based on active modes
                if self.detection_modes['full_frame']:
                    if frame_count % 2 == 0:  # Every 2nd frame for smooth performance
                        class_name, confidence, all_predictions = self.full_frame_classification(frame)
                    
                    if 'class_name' in locals():
                        self.draw_full_frame_results(frame, class_name, confidence, all_predictions)
                
                # Multi-object detection modes
                detections = []
                if any([self.detection_modes['grid'], self.detection_modes['sliding_window'], 
                       self.detection_modes['hybrid']]):
                    if frame_count % 5 == 0:  # Every 5th frame for performance
                        if self.detection_modes['hybrid']:
                            detections = self.hybrid_detection(frame)
                        else:
                            if self.detection_modes['grid']:
                                detections.extend(self.grid_detection(frame))
                            if self.detection_modes['sliding_window']:
                                detections.extend(self.sliding_window_detection(frame))
                            
                            if len(detections) > 0:
                                detections = self.non_max_suppression(detections)
                
                if detections:
                    self.draw_detections(frame, detections)
                
                # Draw UI information
                self.draw_ui_info(frame)
                
                frame_count += 1
            
            # Display the frame
            cv2.imshow('Ultimate Trash Classifier', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                self.detection_modes = {k: False for k in self.detection_modes}
                self.detection_modes['full_frame'] = True
                print("📺 Full Frame Mode")
            elif key == ord('2'):
                self.detection_modes = {k: False for k in self.detection_modes}
                self.detection_modes['grid'] = True
                print("🔲 Grid Detection Mode")
            elif key == ord('3'):
                self.detection_modes = {k: False for k in self.detection_modes}
                self.detection_modes['sliding_window'] = True
                print("🪟 Sliding Window Mode")
            elif key == ord('4'):
                self.detection_modes = {k: False for k in self.detection_modes}
                self.detection_modes['hybrid'] = True
                self.detection_modes['grid'] = True
                self.detection_modes['sliding_window'] = True
                print("🔄 Hybrid Mode")
            elif key == ord('a'):
                self.show_all_predictions = not self.show_all_predictions
                print(f"All predictions: {'ON' if self.show_all_predictions else 'OFF'}")
            elif key == ord('b'):
                self.show_confidence_bars = not self.show_confidence_bars
                print(f"Confidence bars: {'ON' if self.show_confidence_bars else 'OFF'}")
            elif key == ord('r'):
                self.show_detection_regions = not self.show_detection_regions
                print(f"Detection regions: {'ON' if self.show_detection_regions else 'OFF'}")
            elif key == ord('+') or key == ord('='):
                self.confidence_threshold = min(0.95, self.confidence_threshold + 0.05)
                print(f"Confidence threshold: {self.confidence_threshold:.2f}")
            elif key == ord('-'):
                self.confidence_threshold = max(0.05, self.confidence_threshold - 0.05)
                print(f"Confidence threshold: {self.confidence_threshold:.2f}")
            elif key == ord('c'):
                timestamp = int(time.time())
                filename = f"trash_capture_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Captured: {filename}")
            elif key == ord(' '):
                paused = not paused
                print(f"{'⏸️ Paused' if paused else '▶️ Resumed'}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("🔚 Ultimate Trash Classifier stopped")

def main():
    """Main function"""
    print("🗑️ Ultimate Trash Classification System")
    print("=" * 50)
    
    # Check required files
    if not os.path.exists("keras_Model.h5"):
        print("❌ Error: keras_Model.h5 not found!")
        print("Please ensure your Keras model file is in the current directory")
        return
    
    if not os.path.exists("labels.txt"):
        print("❌ Error: labels.txt not found!")
        print("Please ensure your labels file is in the current directory")
        return
    
    # Initialize the ultimate classifier
    classifier = UltimateTrashClassifier()
    
    # Check if initialization was successful
    if not hasattr(classifier, 'model') or classifier.model is None:
        print("❌ Failed to initialize classifier")
        return
    
    # Run the system
    try:
        classifier.run_ultimate_classifier(camera_index=0)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()