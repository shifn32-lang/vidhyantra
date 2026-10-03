"""Django settings for the edutrellis project."""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

FLUX_EDIT_API_URL = os.environ.get('FLUX_EDIT_API_URL', '').strip()
FLUX_EDIT_API_KEY = os.environ.get('FLUX_EDIT_API_KEY', '').strip()

NVIDIA_FLUX_BACKUP_API_KEY = os.environ.get('NVIDIA_FLUX_BACKUP_API_KEY', '').strip()
_flux_backup_key_file = BASE_DIR / '.secrets' / 'nvidia_flux_backup_api_key'
if not NVIDIA_FLUX_BACKUP_API_KEY and _flux_backup_key_file.is_file():
    NVIDIA_FLUX_BACKUP_API_KEY = _flux_backup_key_file.read_text(encoding='utf-8').strip()

NVIDIA_FLUX_DEV_API_KEY = os.environ.get('NVIDIA_FLUX_DEV_API_KEY', '').strip()
_flux_dev_key_file = BASE_DIR / '.secrets' / 'nvidia_flux_dev_api_key'
if not NVIDIA_FLUX_DEV_API_KEY and _flux_dev_key_file.is_file():
    NVIDIA_FLUX_DEV_API_KEY = _flux_dev_key_file.read_text(encoding='utf-8').strip()

QWEN_IMAGE_EDIT_API_URL = os.environ.get('QWEN_IMAGE_EDIT_API_URL', '').strip()
QWEN_IMAGE_EDIT_ENDPOINT_KEY = os.environ.get('QWEN_IMAGE_EDIT_ENDPOINT_KEY', '').strip()
_qwen_image_key_file = BASE_DIR / '.secrets' / 'nvidia_qwen_image_edit_api_key'
NVIDIA_QWEN_IMAGE_EDIT_API_KEY = os.environ.get('NVIDIA_QWEN_IMAGE_EDIT_API_KEY', '').strip()
if not NVIDIA_QWEN_IMAGE_EDIT_API_KEY and _qwen_image_key_file.is_file():
    NVIDIA_QWEN_IMAGE_EDIT_API_KEY = _qwen_image_key_file.read_text(encoding='utf-8').strip()

_kontext_key_file = BASE_DIR / '.secrets' / 'nvidia_flux_kontext_api_key'
NVIDIA_FLUX_KONTEXT_API_KEY = os.environ.get('NVIDIA_FLUX_KONTEXT_API_KEY', '').strip()
if not NVIDIA_FLUX_KONTEXT_API_KEY and _kontext_key_file.is_file():
    NVIDIA_FLUX_KONTEXT_API_KEY = _kontext_key_file.read_text(encoding='utf-8').strip()

_gpt_oss_key_file = BASE_DIR / '.secrets' / 'nvidia_gpt_oss_api_key'
NVIDIA_GPT_OSS_API_KEY = os.environ.get('NVIDIA_GPT_OSS_API_KEY', '').strip()
if not NVIDIA_GPT_OSS_API_KEY and _gpt_oss_key_file.is_file():
    NVIDIA_GPT_OSS_API_KEY = _gpt_oss_key_file.read_text(encoding='utf-8').strip()

SECRET_KEY = os.environ.get(
    'DJANGO_SECRET_KEY',
    'django-insecure-2!^^*c(t)whrn^4w3xkoqx!1p85e5s!-xh0w+xai7&q*80tt@@'
)

DEBUG = True

ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.humanize',
    'myapp.apps.MyappConfig',
    'django.contrib.sitemaps',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'myapp.middleware.CanonicalHostMiddleware',
    'myapp.middleware.SiteDisabledMiddleware',
    'myapp.middleware.PublicAssetCacheMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'myapp.middleware.SingleDeviceSessionMiddleware',
    'myapp.middleware.HideAdminFromNonStaffMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'edutrellis.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'myapp.business_info.context_processor',
                'myapp.views.site_customization_context',
            ],
        },
    },
]

WSGI_APPLICATION = 'edutrellis.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

FILE_UPLOAD_MAX_MEMORY_SIZE = 5 * 1024 * 1024
DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'Asia/Kolkata'
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [BASE_DIR / 'myapp' / 'static']
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

SESSION_COOKIE_AGE = 60 * 60 * 24 * 365
SESSION_SAVE_EVERY_REQUEST = True

GITHUB_OAUTH_CLIENT_ID = os.environ.get('GITHUB_OAUTH_CLIENT_ID', '')
GITHUB_OAUTH_CLIENT_SECRET = os.environ.get('GITHUB_OAUTH_CLIENT_SECRET', '')

SECURE_SSL_REDIRECT = not DEBUG
SESSION_COOKIE_SECURE = not DEBUG
CSRF_COOKIE_SECURE = not DEBUG
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
SECURE_BROWSER_XSS_FILTER = True
X_FRAME_OPTIONS = 'DENY'
SECURE_CONTENT_TYPE_NOSNIFF = True

EMAIL_BACKEND  = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST     = 'smtp.gmail.com'
EMAIL_PORT     = 587
EMAIL_USE_TLS  = True
EMAIL_USE_SSL  = False

EMAIL_TIMEOUT  = 10

EMAIL_HOST_USER      = 'edutrellisprivatelimited@gmail.com'
EMAIL_HOST_PASSWORD  = 'tnjwlzgzmnexoufo'
DEFAULT_FROM_EMAIL   = f'EduTrellis <{EMAIL_HOST_USER}>'
LEAD_RECIPIENT_EMAIL = EMAIL_HOST_USER

TWO_FACTOR_API_KEY = '12feb4c9-9636-11f1-9cb1-0200cd936042'

NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY', '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').strip()
NVIDIA_CHAT_MODEL = os.environ.get('NVIDIA_CHAT_MODEL', 'nvidia/nemotron-3-ultra-550b-a55b').strip()
NVIDIA_FALLBACK_API_KEYS = [
    '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',
    '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',
]
NVIDIA_CHAT_BACKUP_API_KEY = os.environ.get('NVIDIA_CHAT_BACKUP_API_KEY', '').strip()
_chat_backup_key_file = BASE_DIR / '.secrets' / 'nvidia_chat_backup_api_key'
if not NVIDIA_CHAT_BACKUP_API_KEY and _chat_backup_key_file.is_file():
    NVIDIA_CHAT_BACKUP_API_KEY = _chat_backup_key_file.read_text(encoding='utf-8').strip()
NVIDIA_LUNA_API_KEY = os.environ.get(
    'NVIDIA_LUNA_API_KEY',
    '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',
).strip()
_luna_key_file = BASE_DIR / '.secrets' / 'nvidia_luna_api_key'
if not NVIDIA_LUNA_API_KEY and _luna_key_file.is_file():
    NVIDIA_LUNA_API_KEY = _luna_key_file.read_text(encoding='utf-8').strip()
if not NVIDIA_LUNA_API_KEY:
    NVIDIA_LUNA_API_KEY = NVIDIA_CHAT_BACKUP_API_KEY
if NVIDIA_CHAT_BACKUP_API_KEY and NVIDIA_CHAT_BACKUP_API_KEY != NVIDIA_LUNA_API_KEY:
    NVIDIA_FALLBACK_API_KEYS.append(NVIDIA_CHAT_BACKUP_API_KEY)

_nvidia_key_env = os.environ.get('NVIDIA_API_KEYS', '').strip()
if _nvidia_key_env:
    _nvidia_keys = _nvidia_key_env.split(',')
else:
    _nvidia_keys = [NVIDIA_API_KEY] + NVIDIA_FALLBACK_API_KEYS
NVIDIA_API_KEYS = list(dict.fromkeys(k.strip() for k in _nvidia_keys if k.strip()))
NVIDIA_API_KEYS = [k for k in NVIDIA_API_KEYS if k != NVIDIA_LUNA_API_KEY]

_nemotron_super_key_file = BASE_DIR / '.secrets' / 'nvidia_nemotron_super_api_key'
NVIDIA_NEMOTRON_SUPER_API_KEY = os.environ.get(
    'NVIDIA_NEMOTRON_SUPER_API_KEY',
    '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',
).strip()
if not NVIDIA_NEMOTRON_SUPER_API_KEY and _nemotron_super_key_file.is_file():
    NVIDIA_NEMOTRON_SUPER_API_KEY = _nemotron_super_key_file.read_text(encoding='utf-8').strip()
NVIDIA_FLUX_API_KEY = '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'

_terra_key_file = BASE_DIR / '.secrets' / 'nvidia_terra_api_key'
NVIDIA_TERRA_API_KEY = os.environ.get(
    'NVIDIA_TERRA_API_KEY',
    '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',
).strip()
if not NVIDIA_TERRA_API_KEY and _terra_key_file.is_file():
    NVIDIA_TERRA_API_KEY = _terra_key_file.read_text(encoding='utf-8').strip()
NVIDIA_FLUX_EDIT_API_KEY = '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'

TAVILY_API_KEY = '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'

DROPBOX_APP_KEY = 'wgg2fsw5pf16x8q'
DROPBOX_APP_SECRET = '38dg9gi6djz3zuu'
DROPBOX_REFRESH_TOKEN = 'Si57f7yXuB0AAAAAAAAAAZGrsYbd1YLQpvGHxlJES4DRvKr7mDfZo8xqLaJBTY_s'

DROPBOX_IMAGE_ARCHIVE_ENABLED = not (
    'test' in sys.argv
    or os.path.basename(sys.argv[0] if sys.argv else '').startswith(('pytest', 'py.test'))
)

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {'class': 'logging.StreamHandler'},
    },
    'loggers': {
        'myapp': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}

