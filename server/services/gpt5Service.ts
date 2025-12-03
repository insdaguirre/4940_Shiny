import OpenAI from 'openai';
import type { VisionResponse, AnalyzeResponse, Facility } from '../types.js';
import { queryRAG } from './ragService.js';

// Lazy initialization of OpenAI client
function getOpenAIClient() {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error('OPENAI_API_KEY is not set in environment variables');
  }
  return new OpenAI({ apiKey });
}

/**
 * Parses a location string to extract city, region, and country
 * Handles formats like "Ithaca, NY 14850", "San Francisco, CA", "London, UK", etc.
 */
function parseLocation(location: string): {
  city?: string;
  region?: string;
  country?: string;
} {
  const trimmed = location.trim();
  
  // Try to extract components
  // Common patterns:
  // - "City, State ZIP" (US)
  // - "City, State" (US)
  // - "City, Country"
  // - "City, Region, Country"
  // - Just ZIP code
  
  // For US locations, try to extract state
  const usStatePattern = /\b([A-Z]{2})\b/;
  const stateMatch = trimmed.match(usStatePattern);
  
  // Try to split by comma
  const parts = trimmed.split(',').map(p => p.trim());
  
  if (parts.length >= 2) {
    const city = parts[0];
    const secondPart = parts[1];
    
    // Check if second part is a US state (2 letters)
    if (stateMatch && secondPart.length <= 3) {
      return {
        city,
        region: secondPart,
        country: 'US',
      };
    }
    
    // Otherwise, treat as region/country
    if (parts.length >= 3) {
      return {
        city: parts[0],
        region: parts[1],
        country: parts[2],
      };
    }
    
    return {
      city,
      region: secondPart,
    };
  }
  
  // If it's just a ZIP code or single word, return as city
  if (trimmed.length <= 10 && /^\d+/.test(trimmed)) {
    // Likely a ZIP code
    return {
      city: trimmed,
      country: 'US',
    };
  }
  
  return {
    city: trimmed,
  };
}

/**
 * Extracts JSON from response text
 */
function extractJSONFromResponse(content: string): any {
  // Try to find JSON in the response
  const jsonMatch = content.match(/\{[\s\S]*\}/);
  if (jsonMatch) {
    try {
      return JSON.parse(jsonMatch[0]);
    } catch (e) {
      // If parsing fails, try to extract more carefully
    }
  }

  // Fallback: try parsing the entire content
  try {
    return JSON.parse(content);
  } catch (e) {
    throw new Error('Failed to parse JSON from assistant response');
  }
}

/**
 * Analyzes recyclability using Responses API with web search for facilities
 */
export async function analyzeRecyclability(
  visionResult: VisionResponse,
  context: string,
  location: string
): Promise<AnalyzeResponse> {
  try {
    const openai = getOpenAIClient();
    
    // Parse location for user_location parameter
    const parsedLocation = parseLocation(location);
    
    // Build user_location object for web_search
    const userLocation: any = {
      type: 'approximate' as const,
    };
    
    if (parsedLocation.city) {
      userLocation.city = parsedLocation.city;
    }
    if (parsedLocation.region) {
      userLocation.region = parsedLocation.region;
    }
    if (parsedLocation.country) {
      userLocation.country = parsedLocation.country;
    }
    
    // Query RAG service for local regulations
    let ragContext = '';
    let ragSources: string[] = [];
    let ragQueried = false;
    
    // Check if RAG service URL is configured
    const ragServiceUrl = process.env.RAG_SERVICE_URL;
    console.log('RAG Service Status:', {
      configured: !!ragServiceUrl,
      url: ragServiceUrl ? 'set' : 'not set',
      material: visionResult.primaryMaterial,
      location
    });
    
    try {
      const ragResult = await queryRAG(
        visionResult.primaryMaterial,
        location,
        visionResult.condition,
        context
      );
      
      // Track that RAG was queried (even if it returned empty)
      ragQueried = true;
      
      if (ragResult) {
        // Always include sources if available, even if regulations are empty
        ragSources = ragResult.sources || [];
        
        // Use regulations if they exist and are non-empty
        if (ragResult.regulations && ragResult.regulations.trim().length > 0) {
          ragContext = ragResult.regulations;
          console.log('RAG query successful:', {
            regulationsLength: ragResult.regulations.length,
            sourcesCount: ragSources.length,
            sources: ragSources
          });
        } else {
          console.log('RAG query returned empty regulations:', {
            regulationsLength: ragResult.regulations?.length || 0,
            sourcesCount: ragSources.length,
            sources: ragSources,
            material: visionResult.primaryMaterial,
            location
          });
        }
      } else {
        console.log('RAG query returned null - service may be unavailable');
      }
    } catch (error) {
      console.error('RAG query error (non-fatal):', error);
      // Continue without RAG context, but still mark as queried if we attempted it
      if (ragServiceUrl) {
        ragQueried = true;
      }
    }
    
    // Prepare the input prompt
    const input = `You are a recycling and disposal assistant. Analyze this item for recyclability:

Material Analysis:
- Primary Material: ${visionResult.primaryMaterial}
- Category: ${visionResult.category}
- Condition: ${visionResult.condition}
- Contaminants: ${visionResult.contaminants.join(', ') || 'None'}
- Description: ${visionResult.shortDescription}

User Context: ${context || 'None provided'}

User Location: ${location}

${ragContext ? `\nLocal Recycling Regulations (from official sources):\n${ragContext}\n\nUse these official regulations as the primary source for determining recyclability and disposal instructions.` : ''}

Please:
1. Determine if this item is recyclable (true/false)
2. Identify the category it belongs to
3. Determine which bin it should go in (recycling, landfill, compost, hazardous, or unknown)
4. Provide clear, step-by-step instructions for proper disposal
5. Use web search to find 3-5 nearby recycling and disposal facilities in ${location}
6. Return your complete analysis as a JSON object matching this exact structure:
{
  "isRecyclable": boolean,
  "category": string,
  "bin": "recycling" | "landfill" | "compost" | "hazardous" | "unknown",
  "confidence": number (0-1),
  "materialDescription": string (optional),
  "instructions": string[],
  "reasoning": string,
  "locationUsed": string,
  "facilities": [{
    "name": string,
    "type": string,
    "address": string,
    "url": string,
    "notes": string
  }],
  "webSearchSources": string[]
}

IMPORTANT: Include a "webSearchSources" array with URLs of web pages you consulted for recycling information (not just facilities). This should include any websites you used to determine recyclability, disposal methods, or general recycling guidelines. Include the full URLs of the sources you used.

Search for queries like "recycling facilities ${location}" or "${visionResult.primaryMaterial} disposal ${location}" to find local facilities.`;

    // Build tools array with web_search
    const tools: any[] = [
      {
        type: 'web_search' as const,
      },
    ];
    
    // Add user_location if we have valid location data (more than just type)
    if (Object.keys(userLocation).length > 1) {
      tools[0].user_location = userLocation;
    }
    
    // Call Responses API with web_search tool
    const response = await openai.responses.create({
      model: 'gpt-4.1',
      tools: tools,
      input: input,
    });

    // Extract output text from response
    // Responses API returns output_text directly
    const outputText = response.output_text || '';
    
    if (!outputText) {
      throw new Error('No output text from Responses API');
    }

    // Parse JSON from response
    const parsed = extractJSONFromResponse(outputText);

    // Extract web search sources
    const webSearchSources: string[] = Array.isArray(parsed.webSearchSources)
      ? parsed.webSearchSources.filter((url: any) => typeof url === 'string' && url.trim().length > 0)
      : [];

    // Validate and structure the response
    const facilities: Facility[] = Array.isArray(parsed.facilities)
      ? parsed.facilities.map((f: any) => ({
          name: f.name || 'Unknown Facility',
          type: f.type || 'Recycling Center',
          address: f.address || 'Address not available',
          url: f.url || '#',
          notes: f.notes || '',
        }))
      : [];

    // Limit to 5 facilities
    const limitedFacilities = facilities.slice(0, 5);

    return {
      isRecyclable: Boolean(parsed.isRecyclable),
      category: parsed.category || visionResult.category,
      bin: parsed.bin || 'unknown',
      confidence: typeof parsed.confidence === 'number' ? parsed.confidence : 0.5,
      materialDescription: parsed.materialDescription || visionResult.shortDescription,
      instructions: Array.isArray(parsed.instructions) ? parsed.instructions : [],
      reasoning: parsed.reasoning || 'Analysis completed',
      locationUsed: parsed.locationUsed || location,
      facilities: limitedFacilities,
      ragSources: ragSources.length > 0 ? ragSources : undefined,
      ragQueried: ragQueried,
      webSearchSources: webSearchSources.length > 0 ? webSearchSources : undefined,
    };
  } catch (error) {
    console.error('GPT-5 service error:', error);
    throw new Error(
      `Failed to analyze recyclability: ${error instanceof Error ? error.message : 'Unknown error'}`
    );
  }
}
