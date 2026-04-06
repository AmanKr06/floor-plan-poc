import json
import base64
from openai import OpenAI
from src.config import Config

class ComplianceAnalyzer:
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
        # ── STRICT 3-OBJECT VISUAL STANDARDS ──
        self.brand_standards = """
        STRICT FLOOR PLAN STANDARDS v3.0 (VISUAL QA MODE)
        The provided image has been pre-annotated by a deterministic computer vision model. You will see colored bounding boxes labeled 'ZONE', 'DOOR', and 'WINDOW'. 
        
        DO NOT perform any coordinate math. Use your visual understanding of the image to evaluate ONLY the following. 
        *CRITICAL NOTE ON PROXIMITY:* Because walls have thickness and bounding boxes aren't perfect, a DOOR or WINDOW might not physically touch a ZONE box. If a DOOR or WINDOW is immediately adjacent to a ZONE (separated only by a wall or a small visual gap), you MUST count it as servicing that zone.

        1. ZONE EGRESS: Look at every drawn 'ZONE' box. It MUST touch, overlap, contain, or be immediately adjacent to a 'DOOR' box. 
           - Penalty: Deduct exactly 20 points for EACH zone that lacks a door.
           
        2. ZONE ILLUMINATION: Look at every drawn 'ZONE' box. It MUST touch, overlap, contain, or be immediately adjacent to a 'WINDOW' box. 
           - Penalty: Deduct exactly 15 points for EACH zone that lacks a window.
           
        3. PLAN VALIDITY: The overall layout MUST visually contain at least 1 ZONE, 1 DOOR, and 1 WINDOW.
           - Penalty: Deduct exactly 30 points if the entire plan has 0 zones.
           - Penalty: Deduct exactly 20 points if the entire plan has 0 doors.
           - Penalty: Deduct exactly 20 points if the entire plan has 0 windows.
        
        CRITICAL: Base your analysis STRICTLY on the drawn boxes provided in the image. Do not infer or guess about unannotated walls, hallways, or spaces.
        """

    # Helper function to encode the image for OpenAI
    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Notice we added `image_path` to the arguments
    def analyze(self, detections, image_path):
        
        # We just give it the IDs so it knows what text labels to look for on the image
        expected_ids = ", ".join([d['id'] for d in detections])

        prompt = f"""
        You are a senior floor plan compliance analyst.
        Analyze the provided annotated image against the architectural standards. 

        DETECTED LABELS ON IMAGE:
        {expected_ids}

        {self.brand_standards}

        Return ONLY a raw JSON object. No markdown formatting.
        {{
            "compliance_score": <integer 0-100>,
            "status": "Pass" | "Review" | "Fail",
            "explainability_matrix": [
                {{
                    "feature": "<STRICTLY USE THE ID SEEN ON THE IMAGE e.g., ZONE-1>",
                    "issue": "<e.g., Lacks a window>",
                    "point_impact": <integer, e.g., -15>,
                    "reasoning": "<brief explanation based on your visual inspection>"
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