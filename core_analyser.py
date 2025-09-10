import cv2
import numpy as np
import os
import math
import time


class CoreAnalyser:
    """
    The core interactive analysis engine. Measures rotational angle by manually
    selecting points, correcting for geometric and perspective distortions.
    This class is instantiated by a specific analyser (e.g., PictureAnalyser).
    """

    def __init__(self, image, real_catheter_diameter_mm, real_feature_width_mm, filename_prefix="capture"):
        if image is None:
            raise ValueError("Error: Input image cannot be None.")

        # --- Real-world dimensions for correction ---
        self.real_catheter_diameter_mm = real_catheter_diameter_mm
        self.real_feature_width_mm = real_feature_width_mm
        self.real_angular_width = self._calculate_real_angular_width()

        # --- Image and display state ---
        self.original_image = image
        self.display_image = None
        self.window_name = "Manual Catheter Analysis"
        self.filename_prefix = filename_prefix

        # --- Interaction state ---
        self.phase = "ALIGNMENT"
        self.clicked_points = []
        self.zoom_level = 1.0
        self.pan_offset = np.array([0.0, 0.0])
        self.is_panning = False
        self.pan_start = np.array([0, 0])

        self.info_panel_height = 120

    def _calculate_real_angular_width(self):
        """
        Calculates the true angular width of the feature being measured (e.g., electrode or gap)
        on the catheter's surface based on its real-world dimensions. This is a physical constant.
        """
        if self.real_catheter_diameter_mm <= 0 or self.real_feature_width_mm <= 0:
            raise ValueError("Real-world dimensions must be positive values.")
        if self.real_feature_width_mm > self.real_catheter_diameter_mm:
            raise ValueError("Feature width cannot be greater than catheter diameter.")

        real_radius = self.real_catheter_diameter_mm / 2.0
        # The argument to arcsin is (half_width / radius)
        # We multiply by 2 to get the full angular width
        return 2 * math.asin((self.real_feature_width_mm / 2.0) / real_radius)

    def _update_display(self):
        """
        Updates the display window with the current state of the image,
        including zoom, pan, points, and the info panel.
        """
        h, w = self.original_image.shape[:2]

        # --- Create the main display canvas ---
        zoomed_w, zoomed_h = int(w * self.zoom_level), int(h * self.zoom_level)

        # Top-left corner of the zoomed image view
        view_x = int(self.pan_offset[0])
        view_y = int(self.pan_offset[1])

        # Ensure view corners are within the bounds of the scaled image
        view_x = np.clip(view_x, 0, zoomed_w - w)
        view_y = np.clip(view_y, 0, zoomed_h - h)
        self.pan_offset = np.array([float(view_x), float(view_y)])

        # Extract the visible part of the zoomed image
        visible_region = cv2.resize(self.original_image, (zoomed_w, zoomed_h), interpolation=cv2.INTER_LINEAR)
        visible_region = visible_region[view_y:view_y + h, view_x:view_x + w]

        self.display_image = visible_region.copy()

        # --- Draw points on the canvas ---
        for i, (ox, oy) in enumerate(self.clicked_points):
            # Transform original image coordinates to current view coordinates
            vx = int((ox * self.zoom_level) - self.pan_offset[0])
            vy = int((oy * self.zoom_level) - self.pan_offset[1])

            color = (255, 0, 0) if (self.phase == "ALIGNMENT" or i < 2) else (0, 0, 255)
            cv2.circle(self.display_image, (vx, vy), 5, color, -1)

        # --- Create and draw the info panel ---
        panel = np.zeros((self.info_panel_height, self.display_image.shape[1], 3), dtype=np.uint8)
        self._draw_info_text(panel)

        # Combine image and panel
        self.display_image = cv2.vconcat([self.display_image, panel])

        cv2.imshow(self.window_name, self.display_image)

    def _draw_info_text(self, panel):
        """Draws informational text onto the provided panel."""

        def put_text(text, y_pos, color=(255, 255, 255), font_scale=0.6):
            cv2.putText(panel, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)

        put_text(f"PHASE: {self.phase}", 20, color=(0, 255, 255))

        if self.phase == "ALIGNMENT":
            put_text("Click 2 points to define a horizontal line.", 40)
            if len(self.clicked_points) > 0:
                put_text(f"Point 1: {self.clicked_points[0]}", 60)
            if len(self.clicked_points) == 2:
                put_text(f"Point 2: {self.clicked_points[1]}", 80)
                put_text("Confirm alignment? (y/n)", 100, color=(0, 255, 0))

        elif self.phase == "MEASUREMENT":
            instructions = [
                "1. Catheter Top", "2. Catheter Bottom",
                "3. Gap Top Edge", "4. Gap Bottom Edge"
            ]
            put_text(
                f"Click 4 points: {instructions[len(self.clicked_points)] if len(self.clicked_points) < 4 else 'Done'}",
                40)
            coords_str = ", ".join(map(str, self.clicked_points))
            put_text(f"Points: [{coords_str}]", 60)
            put_text("Press 'r' to reset points. 'q' to quit.", 100)

    def _mouse_callback(self, event, x, y, flags, _):
        """Handles all mouse events for panning, zooming, and point selection."""
        # --- Panning ---
        if event == cv2.EVENT_RBUTTONDOWN:
            self.is_panning = True
            self.pan_start = np.array([x, y])
        elif event == cv2.EVENT_RBUTTONUP:
            self.is_panning = False
        elif event == cv2.EVENT_MOUSEMOVE and self.is_panning:
            delta = (self.pan_start - np.array([x, y]))
            self.pan_offset += delta
            self.pan_start = np.array([x, y])

        # --- Zooming ---
        elif event == cv2.EVENT_MOUSEWHEEL:
            zoom_factor = 1.1 if flags > 0 else 1 / 1.1

            # Get mouse position on original image before zoom
            cursor_on_original_x = (self.pan_offset[0] + x) / self.zoom_level
            cursor_on_original_y = (self.pan_offset[1] + y) / self.zoom_level

            self.zoom_level *= zoom_factor
            self.zoom_level = np.clip(self.zoom_level, 1.0, 20.0)

            # Adjust pan offset to keep the same point under the cursor
            self.pan_offset[0] = (cursor_on_original_x * self.zoom_level) - x
            self.pan_offset[1] = (cursor_on_original_y * self.zoom_level) - y

        # --- Point Selection ---
        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.phase == "ALIGNMENT" and len(self.clicked_points) < 2:
                px = int((self.pan_offset[0] + x) / self.zoom_level)
                py = int((self.pan_offset[1] + y) / self.zoom_level)
                self.clicked_points.append((px, py))
            elif self.phase == "MEASUREMENT" and len(self.clicked_points) < 4:
                px = int((self.pan_offset[0] + x) / self.zoom_level)
                py = int((self.pan_offset[1] + y) / self.zoom_level)
                self.clicked_points.append((px, py))

        self._update_display()

    def _align_image(self):
        """Rotates the image to make the line between the two alignment points horizontal."""
        p1, p2 = self.clicked_points
        delta_y = p2[1] - p1[1]
        delta_x = p2[0] - p1[0]
        angle_rad = math.atan2(delta_y, delta_x)
        angle_deg = math.degrees(angle_rad)

        h, w = self.original_image.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

        # Calculate new bounding box to avoid cropping
        cos = np.abs(rot_mat[0, 0])
        sin = np.abs(rot_mat[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust rotation matrix to account for translation
        rot_mat[0, 2] += (new_w / 2) - center[0]
        rot_mat[1, 2] += (new_h / 2) - center[1]

        self.original_image = cv2.warpAffine(self.original_image, rot_mat, (new_w, new_h))

        # Reset state for measurement phase
        self.clicked_points = []
        self.phase = "MEASUREMENT"
        print("Image aligned. Proceeding to measurement phase.")

    def _calculate_and_save_results(self):
        """Performs the final calculation using the angular correction method and saves the image."""
        print("\n--- Starting Final Calculation ---")

        # 1. Apparent Pixel Measurements
        y1, y2 = self.clicked_points[0][1], self.clicked_points[1][1]
        y3, y4 = self.clicked_points[2][1], self.clicked_points[3][1]

        apparent_radius_px = abs(y1 - y2) / 2.0
        catheter_midpoint_y = (y1 + y2) / 2.0

        if apparent_radius_px == 0:
            print("Error: Catheter radius is zero. Cannot divide by zero.")
            return

        offset_app_top = y3 - catheter_midpoint_y
        offset_app_bottom = y4 - catheter_midpoint_y

        # Clamp ratios to avoid math domain errors from slight mis-clicks
        ratio_top = np.clip(offset_app_top / apparent_radius_px, -1.0, 1.0)
        ratio_bottom = np.clip(offset_app_bottom / apparent_radius_px, -1.0, 1.0)

        # 2. Convert Apparent Offsets to Apparent Angles
        angle_app_top_rad = math.asin(ratio_top)
        angle_app_bottom_rad = math.asin(ratio_bottom)

        # 3. Calculate Angular Correction Factor
        angle_app_width_rad = abs(angle_app_top_rad - angle_app_bottom_rad)

        if angle_app_width_rad == 0:
            print("Error: Apparent angular width is zero. Cannot divide by zero.")
            return

        correction_factor_angle = self.real_angular_width / angle_app_width_rad

        # 4. Calculate Final Angle
        angle_app_midpoint_rad = (angle_app_top_rad + angle_app_bottom_rad) / 2.0
        final_angle_rad = angle_app_midpoint_rad * correction_factor_angle
        final_angle_deg = math.degrees(final_angle_rad)

        print(f"Apparent Radius (px): {apparent_radius_px:.2f}")
        print(f"Apparent Angular Width (deg): {math.degrees(angle_app_width_rad):.2f}")
        print(f"Real Angular Width (deg): {math.degrees(self.real_angular_width):.2f}")
        print(f"Angular Correction Factor: {correction_factor_angle:.4f}")
        print(f"Apparent Angle Top (deg): {math.degrees(angle_app_top_rad):.2f}")
        print(f"Apparent Angle Bottom (deg): {math.degrees(angle_app_bottom_rad):.2f}")
        print(f"Apparent Midpoint Angle (deg): {math.degrees(angle_app_midpoint_rad):.2f}")
        print(f"Final Rotation Angle: {final_angle_deg:.2f} degrees")

        # --- Visualization of True Centerline ---
        # Project the final angle back to a corrected pixel offset for visualization
        corrected_offset_px = apparent_radius_px * math.sin(final_angle_rad)
        true_centerline_y = int(catheter_midpoint_y + corrected_offset_px)

        img_width = self.original_image.shape[1]
        cv2.line(self.original_image, (0, int(catheter_midpoint_y)), (img_width, int(catheter_midpoint_y)),
                 (255, 255, 0), 2)  # Cyan
        cv2.line(self.original_image, (0, true_centerline_y), (img_width, true_centerline_y), (0, 255, 255),
                 2)  # Yellow

        # --- Save and Display Final Result ---
        output_dir = "analysis_output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{self.filename_prefix}_analysis.png")
# Create a final image with info panel
        final_panel = np.zeros((self.info_panel_height, self.original_image.shape[1], 3), dtype=np.uint8)
        text = f"Rotation Angle: {final_angle_deg:.2f} degrees"
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_x = (final_panel.shape[1] - w) // 2
        text_y = (self.info_panel_height + h) // 2
        cv2.putText(final_panel, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        final_image = cv2.vconcat([self.original_image, final_panel])

        cv2.imwrite(output_path, final_image)
        print(f"\nAnalysis complete. Saved result to: {output_path}")

        cv2.imshow("Final Result", final_image)
        cv2.waitKey(0)

    def run(self):
        """Main execution loop for the analysis tool."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        self._update_display()

        while True:
            # Check if the user has closed the window
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            key = cv2.waitKey(20) & 0xFF

            if key == ord('q'):
                break

            if self.phase == "ALIGNMENT":
                if len(self.clicked_points) == 2:
                    if key == ord('y'):
                        self._align_image()
                        self._update_display()
                    elif key == ord('n'):
                        self.clicked_points = []
                        self._update_display()

            elif self.phase == "MEASUREMENT":
                if key == ord('r'):
                    self.clicked_points = []
                    self._update_display()

                if len(self.clicked_points) == 4:
                    self._calculate_and_save_results()
                    break

        cv2.destroyAllWindows()


class PictureAnalyser:
    """Handles loading an image from a file and running the analysis."""

    def __init__(self, image_path, real_catheter_diameter_mm, real_feature_width_mm):
        self.image_path = image_path
        self.real_catheter_diameter_mm = real_catheter_diameter_mm
        self.real_feature_width_mm = real_feature_width_mm

    def run(self):
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Error: Image not found at {self.image_path}")

        image = cv2.imread(self.image_path)
        if image is None:
            raise ValueError(f"Error: Could not read image from {self.image_path}")

        base_filename = os.path.basename(self.image_path)
        filename_prefix = os.path.splitext(base_filename)[0]

        analyser = CoreAnalyser(
            image=image,
            real_catheter_diameter_mm=self.real_catheter_diameter_mm,
            real_feature_width_mm=self.real_feature_width_mm,
            filename_prefix=filename_prefix
        )
        analyser.run()


class CameraAnalyser:
    """Handles capturing an image from the camera and running the analysis."""

    def __init__(self, real_catheter_diameter_mm, real_feature_width_mm):
        self.real_catheter_diameter_mm = real_catheter_diameter_mm
        self.real_feature_width_mm = real_feature_width_mm

    def _capture_from_camera(self):
        """Opens a camera feed and captures a still image when 'c' is pressed."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame. Exiting ...")
                break

            display_frame = frame.copy()
            cv2.putText(display_frame, "Press 'c' to capture, 'q' to quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Camera Feed', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return None
            elif key == ord('c'):
                cap.release()
                cv2.destroyAllWindows()
                return frame

        cap.release()
        cv2.destroyAllWindows()
        return None

    def run(self):
        captured_image = self._capture_from_camera()
        if captured_image is not None:
            filename_prefix = f"capture_{int(time.time())}"
            analyser = CoreAnalyser(
                image=captured_image,
                real_catheter_diameter_mm=self.real_catheter_diameter_mm,
                real_feature_width_mm=self.real_feature_width_mm,
                filename_prefix=filename_prefix
            )
            analyser.run()


if __name__ == '__main__':
    # --- DEFINE YOUR REAL-WORLD MEASUREMENTS HERE ---
    REAL_CATHETER_DIAMETER_MM = 1.4
    REAL_ELECTRODE_WIDTH_MM = 0.5

    # --- Calculate the real-world width of the gap between electrodes ---
    if REAL_ELECTRODE_WIDTH_MM * 4 >= REAL_CATHETER_DIAMETER_MM * math.pi:
        raise ValueError("Electrode widths are too large for the given catheter diameter.")

    R_real = REAL_CATHETER_DIAMETER_MM / 2.0
    theta_electrode_rad = REAL_ELECTRODE_WIDTH_MM / R_real
    theta_gap_rad = (2 * math.pi - 4 * theta_electrode_rad) / 4.0
    REAL_GAP_WIDTH_MM = 2 * R_real * math.sin(theta_gap_rad / 2.0)

    print(f"Calculated Real Gap Width: {REAL_GAP_WIDTH_MM:.4f} mm")

    # --- CHOOSE MODE: "CAMERA" or "PICTURE" ---
    MODE = "PICTURE"

    try:
        if MODE == "CAMERA":
            analyser = CameraAnalyser(
                real_catheter_diameter_mm=REAL_CATHETER_DIAMETER_MM,
                real_feature_width_mm=REAL_GAP_WIDTH_MM
            )
            analyser.run()
        elif MODE == "PICTURE":
            # --- IMPORTANT: Set the path to your image file here ---
            image_path = r"C:\Users\admin\PycharmProjects\Alignment_accestment\pictures\BIB15.jpg"  # <-- CHANGE THIS


            analyser = PictureAnalyser(
                image_path=image_path,
                real_catheter_diameter_mm=REAL_CATHETER_DIAMETER_MM,
                real_feature_width_mm=REAL_GAP_WIDTH_MM
            )
            analyser.run()

    except (ValueError, FileNotFoundError) as e:
        print(f"An error occurred: {e}")

