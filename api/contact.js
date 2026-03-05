export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const {
    firstName = '',
    lastName = '',
    email = '',
    phone = '',
    facility = '',
    service = '',
    message = '',
  } = req.body || {};

  if (!firstName || !lastName || !email || !facility || !service || !message) {
    return res.status(400).json({ error: 'Missing required fields' });
  }

  const to = process.env.ZOHO_TO_EMAIL || 'admin@healthbridgesolutions.net';
  const subject = `New Contact Form Message from ${firstName} ${lastName}`;
  const text =
    `Name: ${firstName} ${lastName}\n` +
    `Email: ${email}\n` +
    `Phone: ${phone}\n` +
    `Facility: ${facility}\n` +
    `Service: ${service}\n\n` +
    `Message:\n${message}\n`;

  console.log('Contact form submission:', { to, subject, text });

  return res.status(200).json({ ok: true });
}
