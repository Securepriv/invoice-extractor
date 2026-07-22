export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  
  res.status(200).json({
    status: 'ok',
    groq_configured: !!process.env.GROQ_API_KEY,
    model: 'meta-llama/llama-4-scout-17b-16e-instruct', // ✅ Mis à jour
    platform: 'vercel',
    timestamp: new Date().toISOString()
  });
}
