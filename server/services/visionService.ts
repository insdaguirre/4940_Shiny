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
    const input = `You are an expert recycling and materials classifier.

Task:

You will be given a single image. Your job is to:

1. Identify the main physical item(s) in the image (e.g., "plastic water bottle", "aluminum soda can", "cardboard shipping box").

2. Infer the primary material and any clearly visible secondary materials.

3. Determine a recycling-relevant category and condition.

4. Identify obvious contaminants that would affect recycling (e.g., food residue, labels, liquids, mixed materials).

5. Estimate your confidence on a 0–1 scale.

Important behavior:

- Focus on the **main foreground object** that a user is most likely asking about.

- If multiple items are present, choose the most visually dominant or central item.

- First, internally identify the object type (e.g., "clear PET plastic bottle" or "greasy pizza box"), then map it to materials and category. Do NOT output your reasoning, only the final JSON.

- If you are unsure, choose the best guess but lower the confidence value and use "unknown" / "uncertain" where appropriate.

- Do not invent materials that are not visually supported by the image.

- Use simple, non-technical phrasing for humans (e.g., "clear plastic bottle" instead of "polyethylene terephthalate").

Field semantics:

- primaryMaterial (string): The single most important material by volume/area (e.g., "clear plastic", "aluminum", "cardboard", "glass", "organic waste").

- secondaryMaterials (string[]): Other clearly visible materials (e.g., ["paper label", "plastic cap"]).

- category (string): Recycling-relevant type, such as:

  - "plastic-container", "plastic-film", "metal-can", "glass-bottle", "paper-cardboard",

    "paper-mixed", "textile", "e-waste", "organic-waste", "non-recyclable", "mixed-material"

- condition (string): Short description of cleanliness/shape, e.g.:

  - "clean and empty", "partially full", "heavily soiled with food", "crushed but clean",

    "torn and dirty", "broken glass", "unknown".

- contaminants (string[]): List anything that would interfere with recycling, e.g.:

  - ["food residue", "liquid inside", "tape", "plastic film", "oil/grease", "dirt/soil", "metal staples"].

  - Use [] if there are no obvious contaminants.

- confidence (number): A float in [0, 1] representing overall confidence in your material/category judgment.

- shortDescription (string): 1 short sentence describing what you see in human terms, e.g.:

  - "A clear plastic water bottle with a blue cap, mostly empty."

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

