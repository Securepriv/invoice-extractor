export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');

  if (!process.env.GEMINI_API_KEY) {
    return res.status(500).json({ error: 'GEMINI_API_KEY manquante' });
  }

  try {
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models?key=${process.env.GEMINI_API_KEY}`
    );
    
    const data = await response.json();
    
    if (!response.ok) {
      return res.status(response.status).json({ error: data });
    }

    // Filtrer les modèles qui supportent generateContent
    const generativeModels = data.models
      .filter(m => m.supportedGenerationMethods?.includes('generateContent'))
      .map(m => ({
        name: m.name.replace('models/', ''),
        displayName: m.displayName,
        description: m.description?.substring(0, 100),
        inputTokenLimit: m.inputTokenLimit,
      }));

    return res.status(200).json({
      total: generativeModels.length,
      models: generativeModels.map(m => m.name).sort(),
      details: generativeModels
    });
  } catch (error) {
    return res.status(500).json({ error: error.message });
  }
}
