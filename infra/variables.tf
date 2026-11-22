variable "aws_region" {
  type        = string
  default     = "us-east-1"
  description = "Region for everything, including Bedrock. Pick one where your Claude model is enabled."
}

variable "project" {
  type        = string
  default     = "market-research"
  description = "Name prefix for all resources."
}

variable "desired_count" {
  type        = number
  default     = 1
  description = "Number of running tasks. Set 0 to stop paying for Fargate without destroying anything."
}

variable "image_tag" {
  type        = string
  default     = "latest"
  description = "Image tag in ECR that the service runs."
}

variable "container_cpu" {
  type    = number
  default = 512
}

variable "container_memory" {
  type    = number
  default = 1024
}

variable "bedrock_model_id" {
  type        = string
  default     = "us.anthropic.claude-sonnet-4-20250514-v1:0"
  description = "Bedrock model or inference profile ID. It must be enabled in the Bedrock console (Model access)."
}

variable "alert_email" {
  type        = string
  default     = ""
  description = "Email for the AWS Budget alert. Empty disables the budget."
}

variable "monthly_budget_usd" {
  type    = number
  default = 20
}

variable "github_repo" {
  type        = string
  default     = ""
  description = "owner/name of the GitHub repo allowed to deploy (OIDC). Empty skips creating the deploy role."
}

variable "create_github_oidc_provider" {
  type        = bool
  default     = true
  description = "Set false if this AWS account already has the token.actions.githubusercontent.com OIDC provider."
}
