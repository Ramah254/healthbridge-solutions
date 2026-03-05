import nodemailer from 'nodemailer';

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

  try {
    const transporter = nodemailer.createTransport({
      host: process.env.ZOHO_HOST,
      port: Number(process.env.ZOHO_PORT || 587),
      secure: process.env.ZOHO_SECURE === 'true',
      auth: {
        user: process.env.ZOHO_USER,
        pass: process.env.ZOHO_PASS,
      },
    });

    const to = process.env.ZOHO_TO_EMAIL || process.env.ZOHO_USER;

    const mailOptions = {
      from: `"HealthBridge Website" <${process.env.ZOHO_USER}>`,
      to,
      replyTo: email,
      subject: `New Contact Form Message from ${firstName} ${lastName}`,
      text:
        `Name: ${firstName} ${lastName}\n` +
        `Email: ${email}\n` +
        `Phone: ${phone}\n` +
        `Facility: ${facility}\n` +
        `Service: ${service}\n\n` +
        `Message:\n${message}\n`,
    };

    await transporter.sendMail(mailOptions);

    return res.status(200).json({ ok: true });
  } catch (error) {
    console.error('Zoho mail error:', error);
    return res.status(500).json({ error: 'Failed to send email' });
  }
}
