# SA can read+write objects under the deepemulator/* prefix only.
# Scoped via an IAM condition on resource.name — bucket-level role with object-prefix narrowing.
resource "google_storage_bucket_iam_member" "deepemu_artifacts_rw" {
  bucket = var.artifact_bucket
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.deepemu_runpod.email}"

  condition {
    title       = "deepemulator-prefix-only"
    description = "Restrict object access to the deepemulator/* prefix of the shared artifact bucket."
    expression  = "resource.name.startsWith(\"projects/_/buckets/${var.artifact_bucket}/objects/${var.artifact_prefix}/\")"
  }
}

# SA also needs to list bucket contents (e.g. read latest.txt) — narrowed by the same condition.
resource "google_storage_bucket_iam_member" "deepemu_artifacts_list" {
  bucket = var.artifact_bucket
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.deepemu_runpod.email}"

  condition {
    title       = "deepemulator-prefix-only-read"
    description = "Read access narrowed to the same prefix."
    expression  = "resource.name.startsWith(\"projects/_/buckets/${var.artifact_bucket}/objects/${var.artifact_prefix}/\")"
  }
}
