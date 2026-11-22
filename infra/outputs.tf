output "app_url" {
  value       = "http://${aws_lb.app.dns_name}"
  description = "Open this in a browser once the service is healthy."
}

output "ecr_repository_url" {
  value = aws_ecr_repository.app.repository_url
}

output "secret_id" {
  value       = aws_secretsmanager_secret.keys.name
  description = "Put your API keys into this secret (see infra/README.md)."
}

output "deploy_role_arn" {
  value       = try(aws_iam_role.deploy[0].arn, "")
  description = "Store as the AWS_ROLE_ARN repository secret for the deploy workflow."
}

output "ecs_cluster" {
  value = aws_ecs_cluster.main.name
}

output "ecs_service" {
  value = aws_ecs_service.app.name
}
