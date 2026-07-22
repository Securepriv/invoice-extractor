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
const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-2.0-flash-exp";

// Liste des modèles de fallback
const GEMINI_FALLBACK_MODELS = [
  GEMINI_MODEL,
  "gemini-1.5-flash",
  "gemini-1.5-pro"
];

// ─────────────────────────────────────────────
// Preprocessing d'image avec Sharp
// ─────────────────────────────────────────────
async function preprocessImage(filePath) {
  try {
    const buffer = fs.readFileSync(filePath);
    const metadata = await sharp(buffer).metadata();

    let pipeline = sharp(buffer);

    if (metadata.width > 2048) {
      pipeline = pipeline.resize(2048, null, {
        withoutEnlargement: true,
        fit: 'inside'
      });
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
// Prompt engineering
// ─────────────────────────────────────────────
function buildExtractionPrompt() {
  return `Tu es un système expert d'extraction de données de factures.
Analyse cette image de facture et extrait TOUTES les lignes d'articles du tableau.

RÈGLES STRICTES:
1. Extrait chaque ligne d'article visible dans le tableau de la facture
2. Utilise le POINT comme séparateur décimal (pas la virgule)
3. Les montants doivent être des nombres, pas des chaînes
4. Si une valeur n'est pas lisible, mets null
5. Vérifie que: net_amount ≈ quantity × unit_price (tolérance 0.02)
6. Ne saute aucune ligne, même si elle est partiellement visible
7. Le champ vat_code est souvent 21, 9 ou 0

FORMAT DE SORTIE OBLIGATOIRE (JSON valide uniquement):
{
  "invoice_info": {
    "invoice_number": "4262031",
    "invoice_date": "29/05/26",
    "supplier": "Nom du fournisseur",
    "currency": "EUR",
    "net_amount": 644.53,
    "vat_amount": 135.35,
    "total_amount": 779.88
  },
  "lines": [
    {
      "line_number": 1,
      "quantity": 5.00,
      "unit": "stuks",
      "article_number": "GWA-WMQN02.5K",
      "description": "Watermeter Q3 4 KIWA DN20 G1 L=190 mm",
      "unit_price": 57.31,
      "net_amount": 286.55,
      "vat_code": 21
    }
  ],
  "confidence": 0.95
}

Retourne UNIQUEMENT le JSON, sans texte avant ou après, sans markdown.`;
}

// ─────────────────────────────────────────────
// Parsing JSON robuste
// ─────────────────────────────────────────────
function extractJSON(text) {
  let cleaned = text.trim();
  cleaned = cleaned.replace(/```json\s*/gi, '');
  cleaned = cleaned.replace(/```\s*/g, '');
  cleaned = cleaned.trim();

  try {
    return JSON.parse(cleaned);
  } catch (e) {
    const firstBrace = cleaned.indexOf('{');
    const lastBrace = cleaned.lastIndexOf('}');

    if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
      try {
        return JSON.parse(cleaned.substring(firstBrace, lastBrace + 1));
      } catch (e2) {
        let fixed = cleaned.substring(firstBrace, lastBrace + 1);
        fixed = fixed.replace(/(\d+),(\d+)/g, '$1.$2');
        fixed = fixed.replace(/,\s*([}\]])/g, '$1');

        try {
          return JSON.parse(fixed);
        } catch (e3) {
          return null;
        }
      }
    }
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
    if (typeof line.quantity === 'string') line.quantity = parseFloat(line.quantity.replace(',', '.')) || null;
    if (typeof line.unit_price === 'string') line.unit_price = parseFloat(line.unit_price.replace(',', '.')) || null;
    if (typeof line.net_amount === 'string') line.net_amount = parseFloat(line.net_amount.replace(',', '.')) || null;
    if (typeof line.vat_code === 'string') line.vat_code = parseInt(line.vat_code) || null;

    if (line.quantity && line.unit_price && line.net_amount) {
      const calculated = Math.round(line.quantity * line.unit_price * 100) / 100;
      const diff = Math.abs(calculated - line.net_amount);
      if (diff > 0.02) {
        warnings.push(`Ligne ${index + 1}: ${line.quantity} × ${line.unit_price} = ${calculated} ≠ ${line.net_amount}`);
        line.net_amount_original = line.net_amount;
        line.net_amount = calculated;
        line.auto_corrected = true;
      }
    }

    if (!line.net_amount && line.quantity && line.unit_price) {
      line.net_amount = Math.round(line.quantity * line.unit_price * 100) / 100;
      line.auto_calculated = true;
    }
    if (!line.unit_price && line.quantity && line.net_amount && line.quantity !== 0) {
      line.unit_price = Math.round(line.net_amount / line.quantity * 100) / 100;
      line.auto_calculated = true;
    }

    if (!line.line_number) line.line_number = index + 1;
    return line;
  });

  if (data.invoice_info && data.invoice_info.net_amount) {
    const sumLines = data.lines.reduce((sum, l) => sum + (l.net_amount || 0), 0);
    const totalDiff = Math.abs(sumLines - data.invoice_info.net_amount);
    if (totalDiff > 0.10) {
      warnings.push(`Somme lignes (${sumLines.toFixed(2)}) ≠ Total (${data.invoice_info.net_amount})`);
    }
  }

  return { valid: data.lines.length > 0, errors: [], warnings, data };
}

// ─────────────────────────────────────────────
// Appel Gemini Vision avec retry et fallback
// ─────────────────────────────────────────────
async function callGeminiVision(genAI, base64Image, mimeType, attempt = 1, modelIndex = 0) {
  const currentModel = GEMINI_FALLBACK_MODELS[modelIndex];
  
  console.log(`\n═══════════════════════════════════════`);
  console.log(`🚀 TENTATIVE ${attempt}/${MAX_RETRIES}`);
  console.log(`📦 MODÈLE: ${currentModel}`);
  console.log(`═══════════════════════════════════════`);

  try {
    const model = genAI.getGenerativeModel({
      model: currentModel,
      generationConfig: {
        temperature: 0.1,
        maxOutputTokens: 4096,
        responseMimeType: "application/json",
      }
    });

    const result = await model.generateContent([
      buildExtractionPrompt(),
      {
        inlineData: {
          data: base64Image,
          mimeType: mimeType,
        }
      }
    ]);

    const content = result.response.text();
    console.log(`✅ SUCCÈS avec ${currentModel}`);
    console.log(`📝 Réponse (${content.length} chars): ${content.substring(0, 300)}...`);

    const parsed = extractJSON(content);
    if (!parsed) throw new Error('Impossible de parser le JSON');

    const validation = validateAndCorrectData(parsed);
    if (!validation.valid && attempt < MAX_RETRIES) {
      throw new Error('Données invalides');
    }

    validation.model_used = currentModel;
    return validation;

  } catch (error) {
    const errorMsg = error.message || '';
    console.warn(`❌ ÉCHEC avec ${currentModel}: ${errorMsg}`);

    // Erreur de modèle → essayer le suivant
    const isModelError = 
      errorMsg.includes('not found') ||
      errorMsg.includes('404') ||
      errorMsg.includes('not supported') ||
      errorMsg.includes('permission');

    if (isModelError && modelIndex < GEMINI_FALLBACK_MODELS.length - 1) {
      console.log(`🔄 Fallback vers ${GEMINI_FALLBACK_MODELS[modelIndex + 1]}...`);
      return callGeminiVision(genAI, base64Image, mimeType, 1, modelIndex + 1);
    }

    // Retry temporaire
    if (attempt < MAX_RETRIES && !isModelError) {
      console.warn(`⏳ Retry dans ${attempt}s...`);
      await new Promise(r => setTimeout(r, 1000 * attempt));
      return callGeminiVision(genAI, base64Image, mimeType, attempt + 1, modelIndex);
    }

    throw new Error(`Extraction échouée. Modèles testés: ${GEMINI_FALLBACK_MODELS.slice(0, modelIndex + 1).join(', ')}. Erreur: ${errorMsg}`);
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
      if (err) reject(err);
      else resolve({ fields, files });
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
      error: 'GEMINI_API_KEY non configurée dans Vercel Environment Variables'
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

    console.log(`\n✅ EXTRACTION RÉUSSIE: ${result.data.lines.length} lignes (modèle: ${result.model_used})`);

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
      suggestion: 'Vérifiez que GEMINI_API_KEY est configurée sur https://aistudio.google.com/apikey'
    });

  } finally {
    if (uploadedFile && uploadedFile.filepath) {
      try { fs.unlinkSync(uploadedFile.filepath); } catch (e) { /* ignore */ }
    }
  }
}
