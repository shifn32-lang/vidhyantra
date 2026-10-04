from django.db import migrations

REMOVED_MODEL_KEYS = {
    'gpt-oss-20b', 'reasoning', 'nemotron-3-super', 'flux-kontext-dev', 'qwen-image-edit',
}
REMOVED_KEY_SETTINGS = {
    'NVIDIA_GPT_OSS_API_KEY', 'NVIDIA_FLUX_KONTEXT_API_KEY', 'NVIDIA_QWEN_IMAGE_EDIT_API_KEY',
    'QWEN_IMAGE_EDIT_API_URL', 'QWEN_IMAGE_EDIT_ENDPOINT_KEY',
}


def remove_models(apps, schema_editor):
    AIModelControl = apps.get_model('myapp', 'AIModelControl')
    AIAPIAccess = apps.get_model('myapp', 'AIAPIAccess')
    ProviderAPICredential = apps.get_model('myapp', 'ProviderAPICredential')

    AIModelControl.objects.filter(model_key__in=REMOVED_MODEL_KEYS).delete()
    ProviderAPICredential.objects.filter(setting_name__in=REMOVED_KEY_SETTINGS).delete()
    for access in AIAPIAccess.objects.exclude(model_keys=''):
        keys = [key.strip() for key in access.model_keys.split(',') if key.strip()]
        kept = [key for key in keys if key not in REMOVED_MODEL_KEYS]
        if kept != keys:
            access.model_keys = ','.join(kept)
            access.save(update_fields=['model_keys'])


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0075_voice_calls_and_opencode_activity'),
    ]

    operations = [
        migrations.RunPython(remove_models, migrations.RunPython.noop),
    ]
