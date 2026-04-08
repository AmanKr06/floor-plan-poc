import json
import base64
from openai import OpenAI
from src.config import Config

class ComplianceAnalyzer:
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
        self.brand_standards = """
        STRICT FLOOR PLAN STANDARDS v3.0 (DETERMINISTIC QA MODE)
        You are a senior compliance analyst. You must evaluate the floor plan based strictly on the 'MATHEMATICAL RELATIONSHIPS' provided below. 
        These relationships were calculated by a deterministic geometry engine. Do not second-guess them visually. If the text says a Zone intersects with a Door, it does.
        Also, if you see any object unidentified or wrongly identified by the geometry engine, please take that into consideration as well.

        1. ZONE EGRESS: Every 'ZONE' MUST intersect with at least one 'DOOR'. 
           - Penalty: Deduct exactly 20 points for EACH zone that lacks a door in the relationships list.
           
        2. ZONE ILLUMINATION: Every 'ZONE' MUST intersect with at least one 'WINDOW'. 
           - Penalty: Deduct exactly 15 points for EACH zone that lacks a window in the relationships list.
           
        3. PLAN VALIDITY: The overall layout MUST contain at least 1 ZONE, 1 DOOR, and 1 WINDOW.
           - Penalty: Deduct exactly 30 points if the entire plan has 0 zones.
           - Penalty: Deduct exactly 20 points if the entire plan has 0 doors.
           - Penalty: Deduct exactly 20 points if the entire plan has 0 windows.
        """

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # ── THE MATH ENGINE (Borrowed from our SHAP logic) ──
    def _is_near(self, element, zone, margin=60):
        ex, ey = element['coordinates']['x'], element['coordinates']['y']
        zx, zy = zone['coordinates']['x'], zone['coordinates']['y']
        zw, zh = zone['coordinates']['w'], zone['coordinates']['h']
        
        return (zx - zw/2 - margin) <= ex <= (zx + zw/2 + margin) and \
               (zy - zh/2 - margin) <= ey <= (zy + zh/2 + margin)

    def analyze(self, detections, image_path):
        
        # ── PRE-CALCULATE RELATIONSHIPS ──
        zones = [d for d in detections if d["class_name"] == "zone"]
        doors = [d for d in detections if d["class_name"] == "door"]
        windows = [d for d in detections if d["class_name"] == "window"]

        relationships = []
        for z in zones:
            near_doors = [d['id'] for d in doors if self._is_near(d, z)]
            near_windows = [w['id'] for w in windows if self._is_near(w, z)]
            
            door_text = f"[{', '.join(near_doors)}]" if near_doors else "NONE"
            window_text = f"[{', '.join(near_windows)}]" if near_windows else "NONE"
            
            relationships.append(f"{z['id']} scientifically intersects with -> DOORS: {door_text} | WINDOWS: {window_text}")

        relationships_string = "\n".join(relationships)

        prompt = f"""
        Analyze the floor plan against the architectural standards using the definitive mathematical relationships below.

        MATHEMATICAL RELATIONSHIPS (TRUST THESE EXPLICITLY):
        {relationships_string}

        {self.brand_standards}

        Return ONLY a raw JSON object. No markdown formatting.
        {{
            "compliance_score": <integer 0-100>,
            "status": "Pass" | "Review" | "Fail",
            "explainability_matrix": [
                {{
                    "feature": "<STRICTLY USE THE ID e.g., ZONE-1>",
                    "issue": "<e.g., Lacks a window>",
                    "point_impact": <integer, e.g., -15>,
                    "reasoning": "<brief explanation based on the mathematical relationships provided>"
                }}
            ],
            "summary": "<one sentence assessment>"
        }}
        """

        base64_image = self._encode_image(image_path)

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=800
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": str(e)}