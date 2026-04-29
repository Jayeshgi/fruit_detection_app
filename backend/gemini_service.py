"""
gemini_service.py
------------------------------------------------------------------------------
Handles communication with the Google Gemini API to generate rich
descriptions of detected fruits.

The service takes a fruit name (from the PyTorch model prediction) and
asks Gemini to generate a detailed, informative description including
nutritional info, health benefits, and fun facts.

Includes automatic retry logic for rate limits and a built-in fallback
with real nutritional data for common fruits.
------------------------------------------------------------------------------
"""
import os
import time
import re
from google import genai
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()


# ---------------------------------------------------------------------------
# Built-in nutritional data for common fruits (used when Gemini is unavailable)
# ---------------------------------------------------------------------------
FRUIT_DATA = {
    "apple": {
        "description": "Apples are crisp, juicy fruits with a sweet to tart flavor. They come in many varieties ranging from bright red to green, and are one of the most widely cultivated fruits worldwide.",
        "nutrition": "52 cal, 14g carbs, 2.4g fiber, 4.6mg vitamin C, 0.3g fat per 100g",
        "health_benefits": [
            "Rich in fiber which aids digestion and promotes gut health",
            "Contains antioxidants that may reduce the risk of chronic diseases",
            "Helps regulate blood sugar levels due to low glycemic index",
            "Supports heart health by lowering cholesterol levels",
        ],
        "fun_fact": "There are over 7,500 varieties of apples grown around the world!",
    },
    "banana": {
        "description": "Bananas are elongated, curved fruits with soft, creamy flesh and a thick yellow peel. They have a naturally sweet flavor and are grown in tropical regions around the world.",
        "nutrition": "89 cal, 23g carbs, 2.6g fiber, 8.7mg vitamin C, 358mg potassium per 100g",
        "health_benefits": [
            "Excellent source of potassium which supports heart and muscle function",
            "Provides quick natural energy, ideal before or after exercise",
            "Contains vitamin B6 which helps in brain development and function",
            "High fiber content helps regulate digestion",
        ],
        "fun_fact": "Bananas are technically berries, while strawberries are not!",
    },
    "orange": {
        "description": "Oranges are round citrus fruits with a bright orange rind and juicy, tangy-sweet flesh divided into segments. They are among the most popular fruits globally.",
        "nutrition": "47 cal, 12g carbs, 2.4g fiber, 53mg vitamin C, 0.1g fat per 100g",
        "health_benefits": [
            "Packed with vitamin C which boosts the immune system",
            "Contains flavonoids that have anti-inflammatory properties",
            "High water content helps keep you hydrated",
            "Rich in fiber which promotes healthy digestion",
        ],
        "fun_fact": "The color orange was actually named after the fruit, not the other way around!",
    },
    "strawberry": {
        "description": "Strawberries are small, heart-shaped fruits with bright red skin dotted with tiny seeds. They have a sweet, slightly tangy flavor and are popular fresh, in desserts, and in jams.",
        "nutrition": "32 cal, 7.7g carbs, 2g fiber, 59mg vitamin C, 0.3g fat per 100g",
        "health_benefits": [
            "Very high in vitamin C and manganese",
            "Rich in antioxidants that help fight oxidative stress",
            "May help regulate blood sugar after meals",
            "Contains folate which is important for cell growth",
        ],
        "fun_fact": "Strawberries are the only fruit with seeds on the outside, averaging about 200 seeds per berry!",
    },
    "grape": {
        "description": "Grapes are small, round fruits that grow in clusters on vines. They come in green, red, and purple varieties, with a sweet to mildly tart taste.",
        "nutrition": "69 cal, 18g carbs, 0.9g fiber, 3.2mg vitamin C, 0.2g fat per 100g",
        "health_benefits": [
            "Contain resveratrol which supports heart health",
            "Rich in antioxidants that protect cells from damage",
            "High water content helps with hydration",
            "Provide potassium which helps regulate blood pressure",
        ],
        "fun_fact": "It takes about 2.5 pounds of grapes to make one bottle of wine!",
    },
    "mango": {
        "description": "Mangoes are tropical stone fruits with sweet, juicy orange flesh and a large flat seed. Known as the 'king of fruits', they have a rich, creamy texture and aromatic flavor.",
        "nutrition": "60 cal, 15g carbs, 1.6g fiber, 36mg vitamin C, 0.4g fat per 100g",
        "health_benefits": [
            "Excellent source of vitamins A and C for immune support",
            "Contains digestive enzymes that aid in breaking down food",
            "Rich in folate and B vitamins for energy metabolism",
            "Provides antioxidants that support eye and skin health",
        ],
        "fun_fact": "Mangoes are related to cashews and pistachios - they all belong to the same plant family!",
    },
    "pear": {
        "description": "Pears are bell-shaped fruits with smooth, thin skin and sweet, slightly grainy flesh. They come in varieties like Bartlett, Bosc, and Anjou.",
        "nutrition": "57 cal, 15g carbs, 3.1g fiber, 4.3mg vitamin C, 0.1g fat per 100g",
        "health_benefits": [
            "Very high in dietary fiber which promotes digestive health",
            "Contains copper which supports the nervous system",
            "Low in calories, making it a great snack for weight management",
            "Provides vitamin K which supports bone health",
        ],
        "fun_fact": "Pears ripen from the inside out, which is why they can feel firm outside but be overripe inside!",
    },
    "watermelon": {
        "description": "Watermelon is a large, round or oval fruit with a green rind and sweet, juicy red flesh dotted with black seeds. It is a refreshing summer fruit composed of about 92% water.",
        "nutrition": "30 cal, 7.6g carbs, 0.4g fiber, 8.1mg vitamin C, 0.2g fat per 100g",
        "health_benefits": [
            "Extremely hydrating due to its high water content",
            "Contains lycopene, a powerful antioxidant for heart health",
            "Provides citrulline which may improve exercise performance",
            "Low calorie and refreshing, ideal for summer hydration",
        ],
        "fun_fact": "Watermelon is both a fruit and a vegetable - it belongs to the cucumber and squash family!",
    },
    "lemon": {
        "description": "Lemons are bright yellow citrus fruits with an intensely sour, acidic taste. They are widely used in cooking, beverages, and as a garnish around the world.",
        "nutrition": "29 cal, 9g carbs, 2.8g fiber, 53mg vitamin C, 0.3g fat per 100g",
        "health_benefits": [
            "Extremely rich in vitamin C which boosts immunity",
            "Citric acid may help prevent kidney stones",
            "Contains plant compounds with antioxidant properties",
            "Aids in iron absorption from plant-based foods",
        ],
        "fun_fact": "Lemons contain more sugar than strawberries - the sourness comes from citric acid masking the sweetness!",
    },
    "cherry": {
        "description": "Cherries are small, round stone fruits with glossy red to dark purple skin and sweet or tart flesh. They are popular fresh, in pies, and dried.",
        "nutrition": "50 cal, 12g carbs, 1.6g fiber, 7mg vitamin C, 0.3g fat per 100g",
        "health_benefits": [
            "Rich in anthocyanins which have anti-inflammatory effects",
            "May improve sleep quality due to natural melatonin content",
            "Help reduce muscle soreness after exercise",
            "Contain potassium which supports heart health",
        ],
        "fun_fact": "A cherry tree can produce fruit for up to 100 years!",
    },
    "kiwi": {
        "description": "Kiwis are small, oval fruits with fuzzy brown skin and bright green flesh with tiny black seeds. They have a unique sweet-tart flavor and are native to China.",
        "nutrition": "61 cal, 15g carbs, 3g fiber, 93mg vitamin C, 0.5g fat per 100g",
        "health_benefits": [
            "Contains more vitamin C per gram than most citrus fruits",
            "Rich in vitamin K which supports blood clotting and bone health",
            "Contains actinidin, an enzyme that aids protein digestion",
            "High fiber content supports gut health",
        ],
        "fun_fact": "Kiwis were originally called 'Chinese gooseberries' and were renamed by New Zealand exporters!",
    },
    "peach": {
        "description": "Peaches are soft, round stone fruits with fuzzy skin that ranges from yellow to reddish. They have sweet, aromatic flesh that is juicy when ripe.",
        "nutrition": "39 cal, 10g carbs, 1.5g fiber, 6.6mg vitamin C, 0.3g fat per 100g",
        "health_benefits": [
            "Good source of vitamins A and C for skin and immune health",
            "Contain phenolic compounds with antioxidant properties",
            "Provide dietary fiber that supports digestive health",
            "Low calorie fruit that helps with weight management",
        ],
        "fun_fact": "Peaches are a member of the rose family, along with almonds, plums, and apricots!",
    },
    "pineapple": {
        "description": "Pineapples are large tropical fruits with tough, spiky skin and sweet, tangy yellow flesh. They have a distinctive tropical aroma and are native to South America.",
        "nutrition": "50 cal, 13g carbs, 1.4g fiber, 48mg vitamin C, 0.1g fat per 100g",
        "health_benefits": [
            "Contains bromelain, an enzyme that aids digestion and reduces inflammation",
            "Excellent source of vitamin C and manganese",
            "May boost immunity and suppress inflammation",
            "Rich in antioxidants that fight oxidative stress",
        ],
        "fun_fact": "A pineapple plant takes about 2-3 years to produce a single fruit!",
    },
    "pomegranate": {
        "description": "Pomegranates are round fruits with a thick, leathery red skin containing hundreds of juicy ruby-red seeds called arils. They have a sweet-tart flavor.",
        "nutrition": "83 cal, 19g carbs, 4g fiber, 10mg vitamin C, 1.2g fat per 100g",
        "health_benefits": [
            "Extremely rich in punicalagins, powerful antioxidants",
            "May help lower blood pressure and improve heart health",
            "Has anti-inflammatory properties that reduce chronic disease risk",
            "Contains compounds that may help improve memory",
        ],
        "fun_fact": "A single pomegranate can contain up to 1,400 seeds!",
    },
    "plum": {
        "description": "Plums are smooth-skinned stone fruits that come in many colors including purple, red, yellow, and green. They have juicy, sweet flesh and are enjoyed fresh or dried as prunes.",
        "nutrition": "46 cal, 11g carbs, 1.4g fiber, 9.5mg vitamin C, 0.3g fat per 100g",
        "health_benefits": [
            "Rich in antioxidants, especially polyphenols for bone health",
            "Dried plums (prunes) are well known for relieving constipation",
            "May help lower blood sugar due to their fiber and adiponectin",
            "Contain vitamin C and K for immune and bone support",
        ],
        "fun_fact": "There are over 2,000 varieties of plums grown worldwide!",
    },
    "tomato": {
        "description": "Tomatoes are round, red fruits (often treated as vegetables in cooking) with juicy flesh and a slightly tangy, savory-sweet flavor. They are a staple in cuisines worldwide.",
        "nutrition": "18 cal, 3.9g carbs, 1.2g fiber, 14mg vitamin C, 0.2g fat per 100g",
        "health_benefits": [
            "Rich in lycopene, a powerful antioxidant linked to heart health",
            "Good source of vitamin C, potassium, and folate",
            "May reduce the risk of certain cancers",
            "Supports skin health due to beta-carotene content",
        ],
        "fun_fact": "Tomatoes were once considered poisonous in Europe because they belong to the nightshade family!",
    },
}


def _get_base_fruit_name(predicted_name: str) -> str:
    """
    Extract the base fruit name from the model's prediction.
    The model returns names like 'Apple 8', 'Banana 1', 'Tomato 8' etc.
    This strips the variant number to match our fallback data.
    """
    # Remove trailing numbers and extra whitespace: "Apple 8" -> "apple"
    clean = re.sub(r"\s*\d+\s*$", "", predicted_name).strip().lower()
    # Also handle names like "Apple Braeburn" -> try "apple" first
    words = clean.split()
    return words[0] if words else clean


class GeminiService:
    """
    Wraps the Google Gemini API for generating fruit descriptions.
    Initialized once when the FastAPI server starts.
    Includes retry logic for rate limits and fallback nutritional data.
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 10  # seconds to wait between retries on rate limit

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "paste_your_gemini_api_key_here":
            raise ValueError(
                "GEMINI_API_KEY is not set!\n"
                "1. Copy .env.example to .env\n"
                "2. Paste your Gemini API key in the .env file\n"
                "   Get a key at: https://aistudio.google.com/apikey"
            )

        self.client = genai.Client(api_key=api_key)
        print("[GeminiService] Gemini API configured successfully!")

    async def generate_description(self, fruit_name: str, use_retries: bool = True) -> dict:
        """
        Generate a detailed description of the detected fruit using Gemini.
        
        Args:
            fruit_name:   The name of the fruit predicted by the PyTorch model.
            use_retries:  If True, retry on rate limits (slower but more reliable).
                          If False, try once and fallback instantly (faster response).

        Returns:
            dict with keys: description, nutrition, health_benefits, fun_fact
        """
        prompt = f"""You are a nutrition and fruit expert. A user has uploaded an image 
and our AI model has detected the fruit as: "{fruit_name}".

Please provide the following information about this fruit in a structured way:

1. **Description**: A brief 2-3 sentence description of this fruit (what it looks like, 
   its taste, where it's commonly found).

2. **Nutrition** (per 100g): List the key nutritional values 
   (calories, carbs, fiber, vitamin C, etc.) in a concise format.

3. **Health Benefits**: List 3-4 key health benefits of eating this fruit, 
   each as a single concise sentence.

4. **Fun Fact**: One interesting or surprising fact about this fruit.

IMPORTANT: Respond ONLY in this exact format (no extra text):
DESCRIPTION: <your description here>
NUTRITION: <your nutrition info here>
HEALTH_BENEFITS: <benefit 1> | <benefit 2> | <benefit 3>
FUN_FACT: <your fun fact here>"""

        max_attempts = self.MAX_RETRIES if use_retries else 1

        # Try Gemini
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                )
                text = response.text.strip()
                result = self._parse_response(text, fruit_name)
                print(f"[GeminiService] Description generated successfully (attempt {attempt})")
                return result

            except Exception as e:
                last_error = e
                error_str = str(e)
                # Check if it's a rate limit error (429)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if use_retries and attempt < max_attempts:
                        wait_time = self.RETRY_DELAY * attempt
                        print(
                            f"[GeminiService] Rate limited (attempt {attempt}/{max_attempts}). "
                            f"Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                        continue
                # For non-rate-limit errors, don't retry
                print(f"[GeminiService] Error: {e}")
                break

        # All attempts failed — use built-in fallback data
        print(f"[GeminiService] Gemini unavailable. Using fallback data.")
        return self._get_fallback_data(fruit_name)

    async def refine_prediction(self, image_bytes: bytes, pytorch_predictions: list[dict]) -> str:
        """
        Uses Gemini Vision to look at the image and decide which fruit it actually is.
        This fixes errors where the PyTorch model gets confused by backgrounds.
        """
        # Format the PyTorch guesses to help Gemini (it acts as a hint)
        hints = ", ".join([p["name"] for p in pytorch_predictions])
        
        prompt = f"""You are a world-class expert in pomology (fruit science). 
        
        Our local AI model is confused and suggests these possibilities: {hints}.
        
        YOUR TASK: 
        1. Ignore the hints if they are clearly wrong.
        2. Look at the visual features (shape, stem, texture, color) to identify the fruit.
        3. If it's an Apple, specifically identify which kind if possible, or just say 'Apple'.
        4. Do NOT call an Apple a 'Pear' or vice versa.
        
        Respond with ONLY the name of the fruit (1-3 words max). No extra text."""

        try:
            # Send the raw image to Gemini Vision with temperature 0 for max accuracy
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    prompt,
                    {"mime_type": "image/jpeg", "data": image_bytes}
                ],
                config={"temperature": 0.0}
            )
            
            refined_name = response.text.strip()
            # Remove any trailing punctuation or extra words
            refined_name = re.sub(r"[.!?]$", "", refined_name)
            
            print(f"[GeminiVision] Refined '{pytorch_predictions[0]['name']}' -> '{refined_name}'")
            return refined_name

        except Exception as e:
            print(f"[GeminiVision] Error during refinement: {e}")
            # If vision fails, stick with the PyTorch top prediction
            return pytorch_predictions[0]["name"]

    def _get_fallback_data(self, fruit_name: str) -> dict:
        """
        Return built-in nutritional data for the fruit.
        Uses a dictionary of common fruits with real nutritional information.
        """
        base_name = _get_base_fruit_name(fruit_name)

        if base_name in FRUIT_DATA:
            print(f"[GeminiService] Found fallback data for: {base_name}")
            return FRUIT_DATA[base_name].copy()

        # Generic fallback for unknown fruits
        clean_name = re.sub(r"\s*\d+\s*$", "", fruit_name).strip()
        return {
            "description": f"{clean_name} is a nutritious fruit rich in vitamins and minerals. It can be enjoyed fresh or used in various recipes.",
            "nutrition": "Approximate: 40-60 cal, 10-15g carbs, 1-3g fiber, varies by variety per 100g",
            "health_benefits": [
                "Fruits are naturally low in fat, sodium, and calories",
                "Rich in essential vitamins and minerals for overall health",
                "Good source of dietary fiber for digestive health",
                "Contains antioxidants that help protect cells from damage",
            ],
            "fun_fact": f"Including a variety of fruits like {clean_name} in your diet supports overall health and well-being!",
        }

    def _parse_response(self, text: str, fruit_name: str) -> dict:
        """Parse the structured response from Gemini into a clean dict."""
        result = {
            "description": "",
            "nutrition": "",
            "health_benefits": [],
            "fun_fact": "",
        }

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.upper().startswith("DESCRIPTION:"):
                result["description"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("NUTRITION:"):
                result["nutrition"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("HEALTH_BENEFITS:"):
                benefits_raw = line.split(":", 1)[1].strip()
                result["health_benefits"] = [
                    b.strip() for b in benefits_raw.split("|") if b.strip()
                ]
            elif line.upper().startswith("FUN_FACT:"):
                result["fun_fact"] = line.split(":", 1)[1].strip()

        # Fallback if parsing failed
        if not result["description"]:
            result["description"] = text[:300] if text else f"{fruit_name} is a fruit."

        return result
