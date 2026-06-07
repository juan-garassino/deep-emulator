variable "project_id" {
  description = "GCP project. Canonical home for the deep-* family per ~/Code/CLAUDE.md."
  type        = string
  default     = "garassino-ml"
}

variable "region" {
  description = "GCP region. Workspace standard is europe-west1."
  type        = string
  default     = "europe-west1"
}

variable "artifact_bucket" {
  description = "Shared bucket that holds all garassino-ml artifacts."
  type        = string
  default     = "garassino-ml-artifacts"
}

variable "artifact_prefix" {
  description = "Per-project prefix in the shared bucket. SA write scope is narrowed to this prefix via IAM conditions."
  type        = string
  default     = "deepemulator"
}

variable "sa_account_id" {
  description = "Short ID for the RunPod-to-GCS service account."
  type        = string
  default     = "deepemu-runpod"
}
