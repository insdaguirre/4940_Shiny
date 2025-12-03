// Re-export types from frontend for backend use
export interface AnalyzeRequest {
  image: string;
  location: string;
  context: string;
}

export interface VisionResponse {
  primaryMaterial: string;
  secondaryMaterials: string[];
  category: string;
  condition: string;
  contaminants: string[];
  confidence: number;
  shortDescription: string;
}

export interface Facility {
  name: string;
  type: string;
  address: string;
  url: string;
  notes: string;
  coordinates?: [number, number]; // [longitude, latitude] - optional for MVP
}

export type BinType = 'recycling' | 'landfill' | 'compost' | 'hazardous' | 'unknown';

export interface AnalyzeResponse {
  isRecyclable: boolean;
  category: string;
  bin: BinType;
  confidence: number;
  materialDescription?: string;
  instructions: string[];
  reasoning: string;
  locationUsed: string;
  facilities: Facility[];
  ragSources?: string[];
  ragQueried?: boolean;
  webSearchSources?: string[];
}

