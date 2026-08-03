import argparse
import datetime
import json
import os
import subprocess
import sys


def run_cmd(cmd, capture=False):
    print(f"  $ {cmd}", flush=True)
    if capture:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.returncode
    result = subprocess.run(cmd, shell=True)
    return result.stdout or "", result.returncode


def check_gcloud():
    out, code = run_cmd("gcloud --version", capture=True)
    if code != 0:
        print("ERROR: gcloud CLI not found. Install from https://cloud.google.com/sdk/docs/install")
        sys.exit(1)
    print(f"gcloud OK: {out.split(chr(10))[0]}")


def check_auth():
    out, code = run_cmd("gcloud auth print-access-token --quiet", capture=True)
    if code != 0:
        print("Not authenticated. Run: gcloud auth login")
        sys.exit(1)
    print("Authenticated.")


def build_and_push(project_id, region, tag):
    image = f"{region}-docker.pkg.dev/{project_id}/medicalai/trainer:{tag}"
    print(f"Building image: {image}")

    dockerfile = os.path.join(os.path.dirname(__file__), "Dockerfile")
    context = os.path.dirname(os.path.dirname(__file__))

    cmd = f"gcloud builds submit --tag {image} --project {project_id} --machine-type=e2-highcpu-8"
    out, code = run_cmd(cmd)
    if code != 0:
        print(f"Build failed: {out}")
        sys.exit(1)
    print(f"Image pushed: {image}")
    return image


def submit_job(project_id, region, image_uri, gcs_bucket, service_account, args):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"medicalai-mega-train-{timestamp}"

    config = {
        "displayName": job_name,
        "jobSpec": {
            "workerPoolSpecs": [{
                "machineSpec": {
                    "machineType": args.machine_type,
                    "acceleratorType": args.accelerator,
                    "acceleratorCount": args.accelerator_count,
                },
                "replicaCount": args.replicas,
                "containerSpec": {
                    "imageUri": image_uri,
                    "args": [
                        f"--epochs={args.epochs}",
                        f"--batch_size={args.batch_size}",
                        f"--lr={args.lr}",
                        f"--grad_accum={args.grad_accum}",
                        f"--use_amp={str(args.use_amp).lower()}",
                        f"--freeze_encoders={str(args.freeze_encoders).lower()}",
                        f"--data_dir=/data",
                        f"--model_dir=/model",
                        f"--gcs_bucket={gcs_bucket}",
                        f"--gcs_data_prefix={args.data_prefix}",
                        f"--gcs_model_prefix={args.model_prefix}",
                    ] if not args.max_samples else [
                        f"--epochs={args.epochs}",
                        f"--batch_size={args.batch_size}",
                        f"--lr={args.lr}",
                        f"--grad_accum={args.grad_accum}",
                        f"--use_amp={str(args.use_amp).lower()}",
                        f"--freeze_encoders={str(args.freeze_encoders).lower()}",
                        f"--data_dir=/data",
                        f"--model_dir=/model",
                        f"--gcs_bucket={gcs_bucket}",
                        f"--gcs_data_prefix={args.data_prefix}",
                        f"--gcs_model_prefix={args.model_prefix}",
                        f"--max_samples={args.max_samples}",
                    ],
                    "env": [
                        {"name": "GCS_BUCKET", "value": gcs_bucket},
                        {"name": "TOKENIZERS_PARALLELISM", "value": "false"},
                    ],
                },
            }],
            "scheduling": {"timeout": f"{args.timeout_hours*3600}s", "restartJobOnWorkerRestart": False},
            "serviceAccount": service_account,
        },
    }

    config_path = f"/tmp/vertex_job_{timestamp}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    cmd = (
        f"gcloud ai custom-jobs create "
        f"--region={region} "
        f"--project={project_id} "
        f"--config={config_path} "
        f"--format=json"
    )
    print(f"Submitting Vertex AI job: {job_name}")
    out, code = run_cmd(cmd)
    if code != 0:
        print(f"Job submission failed: {out}")
        sys.exit(1)
    print(f"Job submitted: {job_name}")
    print(f"Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={project_id}")
    os.remove(config_path)


def main():
    parser = argparse.ArgumentParser(description="Submit MedicalAI training to Vertex AI")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--region", default="us-central1", help="GCP region (default: us-central1)")
    parser.add_argument("--bucket", required=True, help="GCS bucket for data and models")
    parser.add_argument("--service-account", required=True, help="Service account email")
    parser.add_argument("--image-tag", default="latest", help="Docker image tag")

    parser.add_argument("--machine-type", default="n1-standard-16")
    parser.add_argument("--accelerator", default="NVIDIA_TESLA_V100")
    parser.add_argument("--accelerator-count", type=int, default=1)
    parser.add_argument("--replicas", type=int, default=1, help="Number of worker replicas (multi-node)")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--freeze-encoders", action="store_true", default=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--data-prefix", default="data/")
    parser.add_argument("--model-prefix", default="models/")
    parser.add_argument("--timeout-hours", type=int, default=24)

    parser.add_argument("--skip-build", action="store_true", help="Skip docker build, use existing image")

    args = parser.parse_args()

    print("=" * 60)
    print("MedicalAI - Vertex AI Training Job Submitter")
    print("=" * 60)

    check_gcloud()
    check_auth()

    if args.skip_build:
        image = f"{args.region}-docker.pkg.dev/{args.project}/medicalai/trainer:{args.image_tag}"
        print(f"Skipping build, using existing: {image}")
    else:
        image = build_and_push(args.project, args.region, args.image_tag)

    submit_job(args.project, args.region, image, args.bucket, args.service_account, args)

    print("\nNext steps:")
    print(f"  1. Watch logs:    gcloud ai custom-jobs stream-logs $(gcloud ai custom-jobs list --region={args.region} --project={args.project} --format='value(name)' --limit=1) --project={args.project}")
    print(f"  2. Download model: gsutil cp -r gs://{args.bucket}/models/ ./checkpoints/vertex_models/")


if __name__ == "__main__":
    main()