from PIL import Image, ImageDraw, ImageFont

class Visualizer:
    COLOR_MAP = {
        "zone": (59, 130, 246),
        "door": (239, 68, 68),
        "window": (245, 158, 11)
    }
    DEFAULT_COLOR = (16, 185, 129)

    @staticmethod
    def draw_bounding_boxes(image_path, detections):
        base_img = Image.open(image_path).convert("RGBA")
        overlay = Image.new("RGBA", base_img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()

        for item in detections:
            name = item['class_name'].lower()
            conf = item['confidence']
            item_id = item.get('id', name.upper())
            
            rgb = Visualizer.COLOR_MAP.get(name, Visualizer.DEFAULT_COLOR)
            box_color = rgb + (255,)
            bg_color = rgb + (160,) # Semi-transparent background

            x, y, w, h = item['coordinates']['x'], item['coordinates']['y'], item['coordinates']['w'], item['coordinates']['h']
            
            # Standard YOLO box coordinates
            x1 = x - (w / 2)
            y1 = y - (h / 2)
            x2 = x + (w / 2)
            y2 = y + (h / 2)

            # Draw the solid bounding box
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
            
            label = f"{item_id} ({conf:.2f})"

            # Calculate text size
            try:
                tb = draw.textbbox((0, 0), label, font=font)
                text_width = tb[2] - tb[0]
                text_height = tb[3] - tb[1]
            except AttributeError:
                text_width, text_height = 100, 15 

            pad_x, pad_y = 4, 2
            
            # STANDARD CV PLACEMENT: Pinned to the top-left corner
            text_x1 = x1
            text_y1 = max(0, y1 - text_height - (pad_y * 2))
            text_x2 = text_x1 + text_width + (pad_x * 2)
            text_y2 = text_y1 + text_height + (pad_y * 2)

            # Draw the background and the text
            draw.rectangle([text_x1, text_y1, text_x2, text_y2], fill=bg_color)
            draw.text((text_x1 + pad_x, text_y1 + pad_y), label, fill=(255, 255, 255, 255), font=font)

        return Image.alpha_composite(base_img, overlay).convert("RGB")