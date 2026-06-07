resource "google_service_account" "deepemu_runpod" {
  account_id   = var.sa_account_id
  display_name = "deepEmulator RunPod GCS sync"
  description  = "Single SA used by RunPod pods to sync training bundles to gs://${var.artifact_bucket}/${var.artifact_prefix}/. Explicit deviation from the 'no SA keys anywhere' workspace rule — RunPod cannot federate via WIF."
}
