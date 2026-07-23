import { GoogleGenerativeAI } from '@google/generative-ai';
import formidable from 'formidable';
import sharp from 'sharp';
import fs from 'fs';

export const config = {
  api: {
    bodyParser: false,
    responseLimit: '20mb',
  },
};

const MAX_RETRIES = parseInt(process.env.MAX_RETRIES) || 3;
const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-flash-latest";

const GEMINI_FALLBACK_MODELS = [
  GEMINI_MODEL,
  "gemini-flash-latest",
  "gemini-2.0-flash-001",
  "gemini-2.5-flash-lite",
  "gemini-flash-lite-latest",
  "gemini-2.0-flash-lite-001"
];

// ─────────────────────────────────────────────
// Preprocessing d'image
// ─────────────────────────────────────────────
async function preprocessImage(filePath) {
  try {
    const buffer = fs.readFileSync(filePath);
    const metadata = await sharp(buffer).metadata();

    let pipeline = sharp(buffer);
    if (metadata.width > 2048) {
      pipeline = pipeline.resize(2048, null, { withoutEnlargement: true, fit: 'inside' });
    }

    const processed = await pipeline
      .normalize()
      .sharpen({ sigma: 1.5, m1: 1.0, m2: 0.5 })
      .modulate({ brightness: 1.05, saturation: 0.1 })
      .gamma(1.8)
      .toFormat('png', { quality: 95 })
      .toBuffer();

    console.log(`✅ Image preprocessée: ${metadata.width}x${metadata.height}`);
    return processed;
  } catch (error) {
    console.warn('⚠️ Preprocessing échoué:', error.message);
    return fs.readFileSync(filePath);
  }
}

// ─────────────────────────────────────────────
// Prompt engineering pour cas complexes
// ─────────────────────────────────────────────
function buildExtractionPrompt() {
  return `Tu es un système EXPERT d'extraction de données de factures complexes.
Analyse cette image et extrait TOUTES les lignes d'articles du tableau.

⚠️ RÈGLES CRITIQUES POUR ÉVITER LES ERREURS JSON:
1. Retourne UNIQUEMENT du JSON valide, PAS de texte, PAS de markdown, PAS de commentaires
2. Utilise TOUJOURS le POINT comme séparateur décimal (89.00, PAS 89,00)
3. Tous les montants doivent être des NOMBRES (89.00), pas des chaînes ("89.00")
4. Échappe correctement les guillemets dans les strings avec \\"
5. N'utilise JAMAIS de retour à la ligne dans une valeur string
6. N'ajoute JAMAIS de virgule finale (trailing comma)

📋 GESTION DES CAS COMPLEXES:

A) NUMÉROS DE SÉRIE (SN:) :
   - Si un article a un numéro de série, ajoute-le à la description
   - Ex: "Polar Breeze Plus Airconditioner SN:001224"

B) REMISES (Korting, Discount, Remise, %) :
   - Si un article a une remise, calcule le prix NET final
   - Ex: 178.00 - 78.32 = 99.68 → net_amount = 99.68
   - Mentionne la remise dans la description: "... Korting 44%"

C) PÉRIODES DE LOCATION :
   - Si une période est indiquée (10-07-2026 t/m 23-07-2026), inclus-la dans la description
   - Ex: "Airconditioner (10-07-2026 t/m 23-07-2026)"

D) DESCRIPTIONS MULTI-LIGNES :
   - Si une description est sur plusieurs lignes visuelles, concatène-les avec un espace

E) LIGNES DE FRAIS SUPPLÉMENTAIRES :
   - Livraison, transport, service : traite-les comme des lignes normales

📐 CALCULS:
- net_amount = prix total NET FINAL après toutes remises éventuelles
- Si tu vois deux montants (prix + remise), fais la soustraction
- Ex: 178.00 (prix) et -78.32 (Korting 44%) → net_amount = 99.68

FORMAT JSON STRICT - RIEN D'AUTRE QUE ÇA:
{
  "lines": [
    {
      "line_number": 1,
      "quantity": 1,
      "article_number": "14665",
      "description": "Polar Breeze Plus Airconditioner SN:001224 (10-07-2026 t/m 23-07-2026) Korting 44%",
      "net_amount": 99.68
    }
  ]
}

⛔ INTERDICTIONS ABSOLUES:
- N'ajoute JAMAIS de texte avant ou après le JSON
- N'utilise JAMAIS de markdown \`\`\`json
- N'ajoute JAMAIS de commentaires // ou /* */
- N'utilise JAMAIS de virgule finale
- Ne mets JAMAIS de valeurs undefined, utilise null

Retourne UNIQUEMENT le JSON valide, commence par { et termine par }.`;
}

// ─────────────────────────────────────────────
// Parsing JSON robuste
// ─────────────────────────────────────────────
function extractJSON(text) {
  if (!text || typeof text !== 'string') return null;
  
  let cleaned = text.trim();
  
  // Retirer markdown
  cleaned = cleaned.replace(/```json\s*/gi, '');
  cleaned = cleaned.replace(/```\s*/g, '');
  cleaned = cleaned.trim();
  
  // Tentative directe
  try {
    return JSON.parse(cleaned);
  } catch (e) {
    console.log('Parse direct échoué, tentative de nettoyage...');
  }
  
  // Extraire entre { et }
  const firstBrace = cleaned.indexOf('{');
  const lastBrace = cleaned.lastIndexOf('}');
  
  if (firstBrace === -1 || lastBrace === -1 || lastBrace <= firstBrace) {
    console.error('Pas de JSON détecté');
    return null;
  }
  
  let extracted = cleaned.substring(firstBrace, lastBrace + 1);
  
  try {
    return JSON.parse(extracted);
  } catch (e) {
    console.log('Parse après extraction échoué, tentative de correction...');
  }
  
  // Corrections agressives
  let fixed = extracted;
  
  // Virgules décimales dans nombres (89,00 → 89.00)
  fixed = fixed.replace(/(\d),(\d{1,2})(?=[,\s\}\]])/g, '$1.$2');
  fixed = fixed.replace(/:\s*(\d+),(\d+)/g, ': $1.$2');
  
  // Virgules finales
  fixed = fixed.replace(/,(\s*[}\]])/g, '$1');
  
  // Commentaires
  fixed = fixed.replace(/\/\/.*$/gm, '');
  fixed = fixed.replace(/\/\*[\s\S]*?\*\//g, '');
  
  // undefined → null
  fixed = fixed.replace(/:\s*undefined/g, ': null');
  
  // Retours à la ligne dans strings
  fixed = fixed.replace(/"([^"]*)\n([^"]*)"/g, '"$1 $2"');
  
  try {
    return JSON.parse(fixed);
  } catch (e) {
    console.log('Parse après correction échoué, tentative finale...');
  }
  
  // Dernier recours : extraire uniquement "lines"
  try {
    const linesMatch = fixed.match(/"lines"\s*:\s*\[([\s\S]*?)\]/);
    if (linesMatch) {
      const linesArray = JSON.parse('[' + linesMatch[1] + ']');
      return { lines: linesArray };
    }
  } catch (e) {
    console.error('Toutes les tentatives de parsing ont échoué');
  }
  
  return null;
}

// ─────────────────────────────────────────────
// Validation des données
// ─────────────────────────────────────────────
function validateAndCorrectData(data) {
  const warnings = [];
  
  if (!data || !data.lines || !Array.isArray(data.lines)) {
    return { valid: false, errors: ['Pas de lignes trouvées'], warnings: [], data: null };
  }

  data.lines = data.lines.map((line, index) => {
    // Convertir strings en nombres
    if (typeof line.quantity === 'string') {
      line.quantity = parseFloat(line.quantity.replace(',', '.')) || null;
    }
    if (typeof line.net_amount === 'string') {
      line.net_amount = parseFloat(line.net_amount.replace(',', '.')) || null;
    }
    
    // Assurer les valeurs par défaut
    if (!line.line_number) line.line_number = index + 1;
    if (!line.quantity) line.quantity = 1;
    if (!line.article_number) line.article_number = '';
    if (!line.description) line.description = '';
    if (line.net_amount == null) line.net_amount = 0;
    
    return line;
  });

  return { 
    valid: data.lines.length > 0, 
    errors: [], 
    warnings, 
    data: { lines: data.lines } 
  };
}

// ─────────────────────────────────────────────
// Appel Gemini avec retry et fallback
// ─────────────────────────────────────────────
async function callGeminiVision(genAI, base64Image, mimeType, attempt = 1, modelIndex = 0) {
  const uniqueModels = [...new Set(GEMINI_FALLBACK_MODELS)];
  const currentModel = uniqueModels[modelIndex];

  if (!currentModel) {
    throw new Error(`Tous les modèles ont été testés: ${uniqueModels.join(', ')}`);
  }
  
  console.log(`\n═══════════════════════════════════════`);
  console.log(`🚀 TENTATIVE ${attempt}/${MAX_RETRIES}`);
  console.log(`📦 MODÈLE: ${currentModel}`);
  console.log(`═══════════════════════════════════════`);

  try {
    const model = genAI.getGenerativeModel({ model: currentModel });

    const result = await model.generateContent({
      contents: [{
        role: "user",
        parts: [
          { text: buildExtractionPrompt() },
          { inlineData: { mimeType: mimeType, data: base64Image } }
        ]
      }],
      generationConfig: {
        temperature: 0.1,
        maxOutputTokens: 4096,
        responseMimeType: "application/json",
      }
    });

    const content = result.response.text();
    console.log(`✅ SUCCÈS avec ${currentModel}`);
    console.log(`📝 Aperçu (${content.length} chars): ${content.substring(0, 300)}...`);

    const parsed = extractJSON(content);
    if (!parsed) {
      console.error('❌ RÉPONSE BRUTE COMPLÈTE:');
      console.error(content);
      throw new Error(`JSON invalide - Réponse: ${content.substring(0, 300)}`);
    }

    const validation = validateAndCorrectData(parsed);
    if (!validation.valid && attempt < MAX_RETRIES) {
      throw new Error('Données invalides');
    }

    validation.model_used = currentModel;
    return validation;

  } catch (error) {
    const errorMsg = error.message || '';
    console.warn(`❌ ÉCHEC avec ${currentModel}: ${errorMsg.substring(0, 200)}`);

    const isQuotaError = 
      errorMsg.includes('429') ||
      errorMsg.includes('Too Many Requests') ||
      errorMsg.includes('quota');

    const isModelError = 
      errorMsg.includes('not found') ||
      errorMsg.includes('404') ||
      errorMsg.includes('not supported') ||
      errorMsg.includes('permission') ||
      errorMsg.includes('is no longer');

    if (isQuotaError && modelIndex < uniqueModels.length - 1) {
      console.log(`⚠️ Quota dépassé, attente 3s puis fallback...`);
      await new Promise(r => setTimeout(r, 3000));
      return callGeminiVision(genAI, base64Image, mimeType, 1, modelIndex + 1);
    }

    if (isModelError && modelIndex < uniqueModels.length - 1) {
      console.log(`🔄 Fallback vers ${uniqueModels[modelIndex + 1]}...`);
      return callGeminiVision(genAI, base64Image, mimeType, 1, modelIndex + 1);
    }

    if (attempt < MAX_RETRIES && !isModelError && !isQuotaError) {
      console.warn(`⏳ Retry dans ${attempt}s...`);
      await new Promise(r => setTimeout(r, 1000 * attempt));
      return callGeminiVision(genAI, base64Image, mimeType, attempt + 1, modelIndex);
    }

    if (isQuotaError) {
      throw new Error(`⏱️ Quota Gemini dépassé. Attendez 1 minute et réessayez.`);
    }

    throw new Error(`Extraction échouée. Modèles testés: ${uniqueModels.slice(0, modelIndex + 1).join(', ')}. Erreur: ${errorMsg.substring(0, 150)}`);
  }
}

// ─────────────────────────────────────────────
// Parser formidable
// ─────────────────────────────────────────────
function parseForm(req) {
  return new Promise((resolve, reject) => {
    const form = formidable({
      maxFileSize: 20 * 1024 * 1024,
      uploadDir: '/tmp',
      keepExtensions: true,
    });
    form.parse(req, (err, fields, files) => {
      if (err) reject(err); else resolve({ fields, files });
    });
  });
}

// ─────────────────────────────────────────────
// Handler principal
// ─────────────────────────────────────────────
export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  if (!process.env.GEMINI_API_KEY) {
    return res.status(500).json({
      success: false,
      error: 'GEMINI_API_KEY non configurée dans Vercel'
    });
  }

  let uploadedFile = null;

  try {
    const { files } = await parseForm(req);
    const fileArray = files.image;
    if (!fileArray) return res.status(400).json({ error: 'Aucune image fournie' });

    uploadedFile = Array.isArray(fileArray) ? fileArray[0] : fileArray;
    console.log(`\n📸 Image reçue: ${uploadedFile.originalFilename} (${(uploadedFile.size / 1024).toFixed(1)} KB)`);

    const processedBuffer = await preprocessImage(uploadedFile.filepath);
    const base64Image = processedBuffer.toString('base64');

    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    const result = await callGeminiVision(genAI, base64Image, 'image/png');

    console.log(`\n✅ EXTRACTION RÉUSSIE: ${result.data.lines.length} lignes (${result.model_used})`);

    return res.status(200).json({
      success: true,
      data: result.data,
      warnings: result.warnings,
      lines_count: result.data.lines.length,
      model_used: result.model_used
    });

  } catch (error) {
    console.error('❌ Erreur globale:', error);
    return res.status(500).json({
      success: false,
      error: error.message,
      suggestion: 'Vérifiez la qualité de l\'image ou attendez si quota dépassé'
    });

  } finally {
    if (uploadedFile && uploadedFile.filepath) {
      try { fs.unlinkSync(uploadedFile.filepath); } catch (e) { /* ignore */ }
    }
  }
}
