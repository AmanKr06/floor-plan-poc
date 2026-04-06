from PIL import Image, ImageDraw, ImageFont

class Visualizer:
    # ─── COLOR MAP (RGB Tuples instead of Hex for Alpha support) ───
    COLOR_MAP = {
        "zone": (59, 130, 246),   # Bright Blue
        "door": (239, 68, 68),    # Crisp Red
        "window": (245, 158, 11)  # Vibrant Orange
    }
    DEFAULT_COLOR = (16, 185, 129) # Fallback Green

    @staticmethod
    def draw_bounding_boxes(image_path, detections):
        """
        Draws bounding boxes with SEMI-TRANSPARENT text backgrounds 
        so UI elements do not blind the GPT-4o Vision model.
        """
        # Convert base image to RGBA to support transparency
        base_img = Image.open(image_path).convert("RGBA")
        
        # Create a blank, transparent overlay image
        overlay = Image.new("RGBA", base_img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        # Try to load font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()

        for item in detections:
            name = item['class_name'].lower()
            conf = item['confidence']
            item_id = item.get('id', name.upper())
            
            # Get base RGB color
            rgb = Visualizer.COLOR_MAP.get(name, Visualizer.DEFAULT_COLOR)
            
            # Create Solid color for the box lines (Alpha 255)
            box_color = rgb + (255,)
            # Create Transparent color for the text background (Alpha 160 = ~60% opaque)
            bg_color = rgb + (160,)

            x, y, w, h = item['coordinates']['x'], item['coordinates']['y'], item['coordinates']['w'], item['coordinates']['h']

            x1 = x - (w / 2)
            y1 = y - (h / 2)
            x2 = x + (w / 2)
            y2 = y + (h / 2)

            # Draw the solid bounding box
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)

            label = f"{item_id} ({conf:.2f})"
            
            # Draw the semi-transparent label background and solid text
            try:
                tb = draw.textbbox((0, 0), label, font=font)
                text_width = tb[2] - tb[0]
                text_height = tb[3] - tb[1]
                text_y1 = max(0, y1 - text_height - 6) 
                
                draw.rectangle([x1, text_y1, x1 + text_width + 6, y1], fill=bg_color)
                draw.text((x1 + 3, text_y1 + 2), label, fill=(255, 255, 255, 255), font=font)
            except AttributeError:
                draw.rectangle([x1, max(0, y1 - 25), x1 + 130, y1], fill=bg_color)
                draw.text((x1 + 3, max(0, y1 - 23)), label, fill=(255, 255, 255, 255), font=font)

        # Merge the transparent overlay with the original image
        final_img = Image.alpha_composite(base_img, overlay).convert("RGB")
        return final_img