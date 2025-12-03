import OpenAI from 'openai';
import type { VisionResponse } from '../types.js';

// Lazy initialization of OpenAI client
function getOpenAIClient() {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error('OPENAI_API_KEY is not set in environment variables');
  }
  return new OpenAI({ apiKey });
}

/**
 * Analyzes an image to identify materials and recycling-relevant conditions
 */
export async function analyzeImage(imageBase64: string): Promise<VisionResponse> {
  // Remove data URL prefix if present
  const base64Data = imageBase64.replace(/^data:image\/\w+;base64,/, '');

  try {
    const openai = getOpenAIClient();
    
    // Prepare input for Responses API
    // Include image as data URL in the input string
    const input = `You are an expert recycling and materials classifier specializing in identifying batteries, electronics, and hazardous materials.

CRITICAL FIRST STEP - Check for Batteries and Electronics:

Before identifying any other material, you MUST check if this is a battery or electronic device. Look for:

- **Battery indicators**: Voltage markings (e.g., "3.7V", "1.5V", "9V"), capacity labels (e.g., "mAh", "Ah"), battery type labels (lithium, alkaline, lead-acid, etc.)
- **Battery terminals**: Metal contacts, positive/negative terminals, connectors
- **Wires attached**: If you see wires connected to a battery-like object, it's a battery
- **Battery shape**: Rectangular, cylindrical, or pouch-shaped objects with voltage/capacity labels
- **Electronic components**: Circuit boards, chips, electronic connectors

**IF YOU SEE ANY BATTERY INDICATORS, the primaryMaterial MUST be "Battery", "Lithium Battery", "Alkaline Battery", or similar battery type. Do NOT identify it as cardboard, plastic, or any other material.**

Task:

You will be given a single image. Your job is to:

1. **FIRST**: Check if this is a battery or electronic device. If battery indicators are present, identify it as a battery.

2. If not a battery, identify the main physical item(s) in the image (e.g., "plastic water bottle", "aluminum soda can", "cardboard shipping box").

3. Infer the primary material and any clearly visible secondary materials.

4. Determine a recycling-relevant category and condition.

5. Identify obvious contaminants that would affect recycling (e.g., food residue, labels, liquids, mixed materials).

6. Estimate your confidence on a 0–1 scale.

Important behavior:

- **Batteries take absolute priority**: If you see ANY battery indicators (voltage, mAh, battery terminals, wires), it's a battery, NOT cardboard, plastic, or paper
- Focus on the **main foreground object** that a user is most likely asking about
- If multiple items are present, choose the most visually dominant or central item
- First, internally identify the object type, then map it to materials and category. Do NOT output your reasoning, only the final JSON
- If you are unsure, choose the best guess but lower the confidence value and use "unknown" / "uncertain" where appropriate
- Do not invent materials that are not visually supported by the image
- Use simple, non-technical phrasing for humans (e.g., "clear plastic bottle" instead of "polyethylene terephthalate")

Field semantics:

- primaryMaterial (string): The single most important material. For batteries, use "Battery", "Lithium Battery", "Alkaline Battery", "Lead-Acid Battery", etc. For other items: "clear plastic", "aluminum", "cardboard", "glass", "organic waste"

- secondaryMaterials (string[]): Other clearly visible materials (e.g., ["paper label", "plastic cap", "wires", "connector"])

- category (string): Recycling-relevant type:
  - For batteries: "e-waste" or "hazardous-waste"
  - For other items: "plastic-container", "plastic-film", "metal-can", "glass-bottle", "paper-cardboard", "paper-mixed", "textile", "organic-waste", "non-recyclable", "mixed-material"

- condition (string): Short description of cleanliness/shape, e.g.:
  - "clean and empty", "partially full", "heavily soiled with food", "crushed but clean", "torn and dirty", "broken glass", "unknown"

- contaminants (string[]): List anything that would interfere with recycling, e.g.:
  - ["food residue", "liquid inside", "tape", "plastic film", "oil/grease", "dirt/soil", "metal staples"]
  - Use [] if there are no obvious contaminants

- confidence (number): A float in [0, 1] representing overall confidence in your material/category judgment

- shortDescription (string): 1 short sentence describing what you see in human terms, e.g.:
  - "A lithium polymer battery with 3.7V and 420mAh capacity, with red and black wires attached"
  - "A clear plastic water bottle with a blue cap, mostly empty"

Output format:

- Return **only** a valid JSON object with exactly these fields:

  { "primaryMaterial": string,
    "secondaryMaterials": string[],
    "category": string,
    "condition": string,
    "contaminants": string[],
    "confidence": number,
    "shortDescription": string }

- No extra text, no markdown, no comments.

Now analyze this image and return the material analysis as JSON:

[Image: data:image/jpeg;base64,${base64Data}]`;

    const response = await openai.responses.create({
      model: 'gpt-4.1',
      input: input,
    });

    const outputText = response.output_text || '';
    if (!outputText) {
      throw new Error('No response from vision API');
    }

    // Parse JSON response
    const parsed = JSON.parse(outputText);
    
    // Validate and return structured response
    return {
      primaryMaterial: parsed.primaryMaterial || 'Unknown',
      secondaryMaterials: Array.isArray(parsed.secondaryMaterials) ? parsed.secondaryMaterials : [],
      category: parsed.category || 'Unknown',
      condition: parsed.condition || 'unknown',
      contaminants: Array.isArray(parsed.contaminants) ? parsed.contaminants : [],
      confidence: typeof parsed.confidence === 'number' ? parsed.confidence : 0.5,
      shortDescription: parsed.shortDescription || parsed.description || 'No description available',
    };
  } catch (error) {
    console.error('Vision API error:', error);
    throw new Error(`Failed to analyze image: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

