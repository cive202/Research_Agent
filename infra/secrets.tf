# The secret container is created here, but its VALUE is set by hand so keys never enter Terraform state:
#   aws secretsmanager put-secret-value --secret-id market-research/api-keys \
#     --secret-string '{"GROQ_API_KEY":"...","TAVILY_API_KEY":""}'
resource "aws_secretsmanager_secret" "keys" {
  name                    = "${var.project}/api-keys"
  recovery_window_in_days = 0 # allow immediate delete on destroy
}
