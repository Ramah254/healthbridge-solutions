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

  const apiKey = process.env.RESEND_API_KEY;

  if (!apiKey) {
    console.error('Missing RESEND_API_KEY');
    return res.status(500).json({ error: 'Email service not configured' });
  }

  const toEmail = 'admin@healthbridgesolutions.net'; // your Zoho inbox

  const subject = `New Contact Form Message from ${firstName} ${lastName}`;

  const text =
    `Name: ${firstName} ${lastName}\n` +
    `Email: ${email}\n` +
    `Phone: ${phone}\n` +
    `Facility: ${facility}\n` +
    `Service: ${service}\n\n` +
    `Message:\n${message}\n`;

  const html = `
    <h2>New Contact Form Submission</h2>
    <p><strong>Name:</strong> ${firstName} ${lastName}</p>
    <p><strong>Email:</strong> ${email}</p>
    <p><strong>Phone:</strong> ${phone}</p>
    <p><strong>Facility:</strong> ${facility}</p>
    <p><strong>Service:</strong> ${service}</p>
    <p><strong>Message:</strong></p>
    <p>${message.replace(/\n/g, '<br>')}</p>
  `;

  try {
    const response = await fetch('https://api.resend.com/emails', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        from: 'HealthBridge Contact <onboarding@resend.dev>',
        to: [toEmail],
        reply_to: email,
        subject,
        text,
        html,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Resend error:', response.status, errorText);
      return res.status(500).json({ error: 'Failed to send email' });
    }

    const data = await response.json();
    console.log('Resend email sent:', data);

    return res.status(200).json({ ok: true });
  } catch (error) {
    console.error('Contact API error:', error);
    return res.status(500).json({ error: 'Failed to process request' });
  }
}
