import requests

from myapp.provider_keys import get_key

TWO_FACTOR_BASE = 'https://2factor.in/API/V1'


def _normalize_phone(phone):
    """Returns a bare 10-digit Indian mobile number from whatever format was
    stored (it may include +91, a leading 0, spaces, etc.)."""
    digits = ''.join(ch for ch in phone if ch.isdigit())
    return digits[-10:]


def send_phone_otp(phone):
    """Sends an auto-generated OTP via 2Factor to `phone`. Returns the
    2Factor session id — must be kept and passed into verify_phone_otp()."""
    number = '91' + _normalize_phone(phone)
    url = f'{TWO_FACTOR_BASE}/{get_key("TWO_FACTOR_API_KEY")}/SMS/{number}/AUTOGEN'
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get('Status') != 'Success':
        raise RuntimeError(f'2Factor send failed: {data}')
    return data['Details']


def verify_phone_otp(session_id, otp):
    """True if `otp` matches what 2Factor sent for that session."""
    url = f'{TWO_FACTOR_BASE}/{get_key("TWO_FACTOR_API_KEY")}/SMS/VERIFY/{session_id}/{otp}'
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get('Status') == 'Success'
