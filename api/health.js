export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  
  res.status(200).json({
    status: 'ok',
    groq_configured: !!process.env.GROQ_API_KEY,
    model: 'llama-3.2-90b-vision-preview',
    platform: 'vercel',
    timestamp: new Date().toISOString()
  });
}
