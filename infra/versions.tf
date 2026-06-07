terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  # Workspace-wide remote state bucket, per ~/Code/CLAUDE.md § GCP architecture.
  backend "gcs" {
    bucket = "garassino-op-tf-state"
    prefix = "deepemulator"
  }
}
