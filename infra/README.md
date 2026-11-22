# AWS deployment (optional)

Runs the app on **ECS Fargate behind an Application Load Balancer**. The assignment does not require this; the app
runs locally with `streamlit run app.py`.

> **Status: written but not applied.** I could not run `terraform validate` or `apply` while building this (no
> Terraform or AWS account in the build environment). Run `terraform validate` first and read the plan before applying.

## What it creates

VPC with two public subnets (no NAT gateway, to save about $30/month), ALB, ECS cluster and Fargate service, ECR repo,
CloudWatch log group, a Secrets Manager secret for the API keys, IAM roles (the task role may only call Bedrock), an
optional budget alert and an optional GitHub OIDC deploy role.

**Cost:** roughly $30 to $40 per month while running (ALB about $17 to $20, one small Fargate task about $15).
Set `desired_count = 0` to stop the Fargate cost, or `terraform destroy` to remove everything.

## First deploy

```bash
cd infra
terraform init
terraform validate
terraform apply -var 'alert_email=you@example.com'      # optional: budget alert; add -var 'github_repo=owner/name' for CI deploys
```

1. Put your keys in the secret (they never enter Terraform state):
   ```bash
   aws secretsmanager put-secret-value --secret-id market-research/api-keys \
     --secret-string '{"GROQ_API_KEY":"your-key","TAVILY_API_KEY":""}'
   ```
2. Push the first image (the service stays unhealthy until an image exists):
   ```bash
   REPO=$(terraform output -raw ecr_repository_url)
   aws ecr get-login-password | docker login --username AWS --password-stdin "${REPO%%/*}"
   docker build -t "$REPO:latest" .. && docker push "$REPO:latest"
   aws ecs update-service --cluster market-research --service market-research --force-new-deployment
   ```
3. Open `terraform output app_url`.

## Deploy from GitHub Actions

Apply with `-var 'github_repo=owner/name'`, then add the repository secret `AWS_ROLE_ARN` (value: `terraform output
deploy_role_arn`) and run the **Deploy to AWS** workflow manually. It uses OIDC, so no AWS keys are stored in GitHub.

## Using Claude on Bedrock

In the AWS console open **Bedrock, Model access** and enable a Claude model in your region. In the sidebar choose
**bedrock**, or set `LLM_PROVIDER=bedrock` and `BEDROCK_MODEL_ID` (the default is a cross-region inference profile ID,
so check it against what your account lists). The task role already allows `bedrock:InvokeModel`. Locally, run
`pip install boto3` and `aws configure`.

## Known limits

HTTP only (add an ACM certificate and an HTTPS listener for production); no authentication in front of the app, so
anyone with the URL can spend your LLM quota; state is not shared between tasks, so keep `desired_count = 1`.

## Tear down

```bash
terraform destroy
```
