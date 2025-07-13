#!/bin/bash
# ECS Deployment Script for LangChain Q&A Application

set -e  # Exit on error

# Configuration
AWS_REGION=${AWS_REGION:-"us-east-1"}
AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID}
ECR_REPOSITORY=${ECR_REPOSITORY:-"langchain-qa"}
ECS_CLUSTER=${ECS_CLUSTER:-"langchain-qa-cluster"}
ECS_SERVICE=${ECS_SERVICE:-"langchain-qa-service"}
ENVIRONMENT=${ENVIRONMENT:-"production"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Validate required environment variables
validate_env() {
    if [ -z "$AWS_ACCOUNT_ID" ]; then
        print_error "AWS_ACCOUNT_ID is not set"
        exit 1
    fi
}

# Build Docker image
build_image() {
    print_status "Building Docker image..."
    docker build -t ${ECR_REPOSITORY}:${BUILD_TAG} .
    
    # Tag for latest
    docker tag ${ECR_REPOSITORY}:${BUILD_TAG} ${ECR_REPOSITORY}:latest
}

# Push to ECR
push_to_ecr() {
    print_status "Logging in to ECR..."
    aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
    
    print_status "Pushing image to ECR..."
    docker tag ${ECR_REPOSITORY}:${BUILD_TAG} ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${BUILD_TAG}
    docker tag ${ECR_REPOSITORY}:latest ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:latest
    
    docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${BUILD_TAG}
    docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:latest
}

# Update ECS Task Definition
update_task_definition() {
    print_status "Updating task definition..."
    
    # Replace variables in task definition
    export ECR_REPOSITORY_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}"
    export IMAGE_TAG="${BUILD_TAG}"
    export APP_VERSION="${BUILD_TAG}"
    
    # Use envsubst to replace variables
    envsubst < task-definition.json > task-definition-${BUILD_TAG}.json
    
    # Register new task definition
    TASK_DEFINITION_ARN=$(aws ecs register-task-definition \
        --cli-input-json file://task-definition-${BUILD_TAG}.json \
        --region ${AWS_REGION} \
        --query 'taskDefinition.taskDefinitionArn' \
        --output text)
    
    print_status "Registered task definition: ${TASK_DEFINITION_ARN}"
}

# Update ECS Service
update_service() {
    print_status "Updating ECS service..."
    
    aws ecs update-service \
        --cluster ${ECS_CLUSTER} \
        --service ${ECS_SERVICE} \
        --task-definition ${TASK_DEFINITION_ARN} \
        --region ${AWS_REGION} \
        --force-new-deployment
    
    print_status "Service update initiated"
}

# Wait for deployment to complete
wait_for_deployment() {
    print_status "Waiting for deployment to complete..."
    
    aws ecs wait services-stable \
        --cluster ${ECS_CLUSTER} \
        --services ${ECS_SERVICE} \
        --region ${AWS_REGION}
    
    print_status "Deployment completed successfully!"
}

# Main deployment flow
main() {
    print_status "Starting deployment for LangChain Q&A Application"
    
    # Generate build tag
    export BUILD_TAG=$(date +%Y%m%d%H%M%S)-$(git rev-parse --short HEAD 2>/dev/null || echo "local")
    print_status "Build tag: ${BUILD_TAG}"
    
    validate_env
    build_image
    push_to_ecr
    update_task_definition
    update_service
    wait_for_deployment
    
    print_status "Deployment completed! Build tag: ${BUILD_TAG}"
}

# Run main function
main "$@"