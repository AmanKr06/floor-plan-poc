from PIL import Image, ImageDraw, ImageFont

class Visualizer:
    # ─── NEW: COLOR DICTIONARY ───
    # We assign distinct, high-contrast hex colors to each specific class
    COLOR_MAP = {
        "zone": "#3B82F6",   # Bright Blue
        "door": "#EF4444",   # Crisp Red
        "window": "#F59E0B"  # Vibrant Orange
    }
    DEFAULT_COLOR = "#10B981" # Fallback Green

    @staticmethod
    def draw_bounding_boxes(image_path, detections):
        """
        Takes the original image and the list of detected elements,
        and draws distinct, color-coded bounding boxes around them.
        """
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Try to load a slightly larger, clearer font
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except IOError:
            font = ImageFont.load_default()

        for item in detections:
            name = item['class_name'].lower()
            conf = item['confidence']
            item_id = item.get('id', name.upper())
            
            # ─── MAP THE COLOR ───
            # Look up the class name in our dictionary. If it doesn't exist, use green.
            box_color = Visualizer.COLOR_MAP.get(name, Visualizer.DEFAULT_COLOR)

            x, y, w, h = item['coordinates']['x'], item['coordinates']['y'], item['coordinates']['w'], item['coordinates']['h']

            # Convert center coordinates to Top-Left / Bottom-Right
            x1 = x - (w / 2)
            y1 = y - (h / 2)
            x2 = x + (w / 2)
            y2 = y + (h / 2)

            # Draw the bounding box (Increased width from 3 to 4 for visibility)
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=4)

            label = f"{item_id} ({conf:.2f})"
            
            # Draw the label text background cleanly ON TOP of the box line
            try:
                tb = draw.textbbox((0, 0), label, font=font)
                text_width = tb[2] - tb[0]
                text_height = tb[3] - tb[1]
                
                # Calculate Y position so the text box sits right above the bounding box
                text_y1 = max(0, y1 - text_height - 6) 
                
                draw.rectangle([x1, text_y1, x1 + text_width + 6, y1], fill=box_color)
                draw.text((x1 + 3, text_y1 + 2), label, fill="white", font=font)
            except AttributeError:
                # Fallback for older PIL versions
                draw.rectangle([x1, max(0, y1 - 25), x1 + 130, y1], fill=box_color)
                draw.text((x1 + 3, max(0, y1 - 23)), label, fill="white", font=font)

        return img