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

# storage.objects.list authorizes against the BUCKET resource
# (projects/_/buckets/<name>), which never matches an objects/<prefix>
# condition — a conditioned viewer role grants nothing for listing, and
# gcs.download_dir() (corpus/bundle staging) 403s without list.
# legacyBucketReader = objects.list + buckets.get only. Trade-off: the SA can
# see object NAMES bucket-wide (not contents — reads stay prefix-scoped via
# the conditioned objectAdmin above). Accepted; documented in CLAUDE.md.
resource "google_storage_bucket_iam_member" "deepemu_artifacts_list" {
  bucket = var.artifact_bucket
  role   = "roles/storage.legacyBucketReader"
  member = "serviceAccount:${google_service_account.deepemu_runpod.email}"
}
