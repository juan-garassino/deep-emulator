# SA key for RunPod (the explicit "no SA keys" deviation).
# After `terraform apply`, run `make tf_output_sa_key` to write the JSON to
# /tmp/gcp-sa-deepemu.json (gitignored), then upload that file once to RunPod as
# Pod-secret named `gcp-sa-deepemu`. Rotate by `terraform taint
# google_service_account_key.deepemu_runpod_key && make tf_apply`.
resource "google_service_account_key" "deepemu_runpod_key" {
  service_account_id = google_service_account.deepemu_runpod.name
  public_key_type    = "TYPE_X509_PEM_FILE"
}
