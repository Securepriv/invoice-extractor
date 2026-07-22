import Groq from 'groq-sdk';

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');

  if (!process.env.GROQ_API_KEY) {
    return res.status(500).json({ error: 'GROQ_API_KEY manquante' });
  }

  try {
    const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
    const models = await groq.models.list();
    
    // Filtrer uniquement les modèles vision
    const visionModels = models.data.filter(m => 
      m.id.toLowerCase().includes('vision') || 
      m.id.toLowerCase().includes('scout') ||
      m.id.toLowerCase().includes('maverick') ||
      m.id.toLowerCase().includes('llama-4')
    );

    return res.status(200).json({
      total_models: models.data.length,
      vision_models: visionModels.map(m => m.id),
      all_models: models.data.map(m => m.id).sort()
    });
  } catch (error) {
    return res.status(500).json({ error: error.message });
  }
}
