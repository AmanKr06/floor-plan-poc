from PIL import Image, ImageDraw, ImageFont

class Visualizer:
    @staticmethod
    def draw_bounding_boxes(image_path, detections):
        """
        Takes the original image and the list of detected elements,
        and draws clean bounding boxes around them.
        """
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Try to load a default font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()

        for item in detections:
            name = item['class_name']
            conf = item['confidence']
            x, y, w, h = item['coordinates']['x'], item['coordinates']['y'], item['coordinates']['w'], item['coordinates']['h']

            # YOLO gives us the center coordinates. We need to convert them 
            # to Top-Left (x1, y1) and Bottom-Right (x2, y2) for drawing.
            x1 = x - (w / 2)
            y1 = y - (h / 2)
            x2 = x + (w / 2)
            y2 = y + (h / 2)

            # Draw the bounding box (Green for visibility)
            draw.rectangle([x1, y1, x2, y2], outline="#10B981", width=3)

            # Draw the label text background and text
            label = f"{name} {conf:.2f}"
            
            # Simple fallback for text background size if textbbox fails
            try:
                tb = draw.textbbox((x1, y1), label, font=font)
                draw.rectangle([x1, max(0, y1 - 20), tb[2], y1], fill="#10B981")
            except AttributeError:
                draw.rectangle([x1, max(0, y1 - 20), x1 + 100, y1], fill="#10B981")

            draw.text((x1 + 2, max(0, y1 - 20)), label, fill="white", font=font)

        return img