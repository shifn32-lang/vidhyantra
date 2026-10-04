Uploaded image editing
======================

FLUX generation backup
----------------------

If FLUX.2 Klein text-to-image generation fails due to an unavailable service,
credential error, rate limit, network error, or invalid output, generation
retries with a second Klein credential first, then once on NVIDIA FLUX.1-dev.
Set NVIDIA_FLUX_BACKUP_API_KEY or use the ignored
.secrets/nvidia_flux_backup_api_key file for the second Klein credential.
Set NVIDIA_FLUX_DEV_API_KEY in the deployment
environment or put the key in the ignored .secrets/nvidia_flux_dev_api_key file.
The backup uses its own credential and returns 1024x1024 images. It does not
accept uploaded photos, bypass content refusals, or back up other image providers.
The selected model label is retained in existing usage records, so these records
do not identify which backend actually completed a fallback request.

Reference: https://docs.api.nvidia.com/nim/reference/black-forest-labs-flux_1-dev-infer

Upload configuration
--------------------

Pillow already normalizes uploaded PNG, JPEG, and WebP images: EXIF orientation,
proportional resizing to 1024 pixels, transparency flattening, and JPEG encoding.
Edit instructions with an attachment route automatically to FLUX; the selected
ChatGPT 5.6 name is retained in response headers and saved messages.

The hosted NVIDIA FLUX preview accepts only preset example images. Conversion
cannot make it accept an arbitrary upload:
https://docs.api.nvidia.com/nim/reference/black-forest-labs-flux_2-klein-4b-infer

To enable actual uploads, deploy an upload-capable FLUX.2 Klein NIM and set
FLUX_EDIT_API_URL to its full inference URL. Set FLUX_EDIT_API_KEY only if that
deployment requires bearer authentication. Restart Django after configuration.
The adapter sends the existing FLUX JSON request with an image array containing
a JPEG data URI and expects the NIM artifacts/base64 response. Normal image
generation continues to use the hosted endpoint. Hosted credentials are never
forwarded to the custom endpoint.

Deployment documentation:
https://docs.nvidia.com/nim/visual-genai/latest/getting-started.html

An upload-capable deployment has not been configured or live-tested here.
