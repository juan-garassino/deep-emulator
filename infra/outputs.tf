output "sa_email" {
  description = "Service account that RunPod uses to write artifacts."
  value       = google_service_account.deepemu_runpod.email
}

output "artifact_prefix_uri" {
  description = "Canonical GCS prefix for deepEmulator artifacts."
  value       = "gs://${var.artifact_bucket}/${var.artifact_prefix}"
}

output "sa_key_json" {
  description = "Service-account key as JSON. Sensitive — surfaced for upload to RunPod, never commit."
  value       = base64decode(google_service_account_key.deepemu_runpod_key.private_key)
  sensitive   = true
}
