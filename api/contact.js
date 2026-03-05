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
    console.log('Contact form submission:', {
      firstName,
      lastName,
      email,
      phone,
      facility,
      service,
      message,
    });

    return res.status(200).json({ ok: true });
  } catch (error) {
    console.error('Contact API error:', error);
    return res.status(500).json({ error: 'Failed to process request' });
  }
}
