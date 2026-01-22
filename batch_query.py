"""
Bedrock Batch Inference Template
================================
A reusable template for running batch LLM/VLM inference on AWS Bedrock.

Usage:
1. Define your prompt formatting function
2. Prepare your input data
3. Use the helper functions to create batch input and submit jobs

Example:
    # 1. Define how to format your prompts
    def format_my_prompt(record: dict) -> str:
        return f"Summarize this: {record['text']}"
    
    # 2. Create batch input records
    records = [{"id": "1", "text": "Hello world"}, ...]
    batch_inputs = [
        create_llm_batch_input(
            prompt=format_my_prompt(r),
            record_id=r["id"]
        ) for r in records
    ]
    
    # 3. Save to JSONL and upload to S3
    save_batch_input_jsonl(batch_inputs, "my_batch.jsonl")
    upload_jsonl_in_chunks("my_batch.jsonl", "my-bucket", "inputs/")
    
    # 4. Submit batch jobs
    manager = BedrockBatchJobManager(...)
    manager.process_file_chunks(chunk_files, job_name_prefix="my-job")
"""

import boto3
import json
import time
import io
import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Callable, Any
import logging
from dataclasses import dataclass
import queue
import threading
import warnings

# Optional: for S3 filesystem operations
try:
    import s3fs
except ImportError:
    s3fs = None


# =============================================================================
# PART 1: PROMPT FORMATTING & BATCH INPUT CREATION
# =============================================================================

def create_llm_batch_input(
    prompt: str,
    record_id: Optional[str] = None,
    max_tokens: int = 1024,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    stop_sequences: Optional[List[str]] = None,
) -> dict:
    """
    Create a single batch input record for text-only LLM inference.
    
    Args:
        prompt: The user prompt text
        record_id: Unique identifier for tracking this record in output
        max_tokens: Maximum tokens to generate (default: 1024)
        system_prompt: Optional system prompt
        temperature: Sampling temperature (0.0-1.0)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        stop_sequences: List of stop sequences
    
    Returns:
        Dictionary formatted for Bedrock batch inference
    """
    model_input = {
        "max_tokens": max_tokens,
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    }
    
    # Add optional parameters
    if system_prompt:
        model_input["system"] = system_prompt
    if temperature is not None:
        model_input["temperature"] = temperature
    if top_p is not None:
        model_input["top_p"] = top_p
    if top_k is not None:
        model_input["top_k"] = top_k
    if stop_sequences:
        model_input["stop_sequences"] = stop_sequences
    
    record = {"modelInput": model_input}
    
    if record_id:
        record["recordId"] = record_id
    
    return record


def create_vlm_batch_input(
    prompt: str,
    image_data: str,
    media_type: str = "image/jpeg",
    record_id: Optional[str] = None,
    max_tokens: int = 1024,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    stop_sequences: Optional[List[str]] = None,
) -> dict:
    """
    Create a single batch input record for Vision-Language Model (VLM) inference.
    
    Args:
        prompt: The user prompt text
        image_data: Base64-encoded image data
        media_type: Image MIME type (default: "image/jpeg")
        record_id: Unique identifier for tracking this record in output
        max_tokens: Maximum tokens to generate (default: 1024)
        system_prompt: Optional system prompt
        temperature: Sampling temperature (0.0-1.0)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        stop_sequences: List of stop sequences
    
    Returns:
        Dictionary formatted for Bedrock batch inference
    """
    model_input = {
        "max_tokens": max_tokens,
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    }
    
    # Add optional parameters
    if system_prompt:
        model_input["system"] = system_prompt
    if temperature is not None:
        model_input["temperature"] = temperature
    if top_p is not None:
        model_input["top_p"] = top_p
    if top_k is not None:
        model_input["top_k"] = top_k
    if stop_sequences:
        model_input["stop_sequences"] = stop_sequences
    
    record = {"modelInput": model_input}
    
    if record_id:
        record["recordId"] = record_id
    
    return record


def create_multi_turn_batch_input(
    messages: List[Dict[str, Any]],
    record_id: Optional[str] = None,
    max_tokens: int = 1024,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    stop_sequences: Optional[List[str]] = None,
) -> dict:
    """
    Create a batch input record for multi-turn conversations.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
                  e.g., [{"role": "user", "content": "Hello"}, 
                         {"role": "assistant", "content": "Hi!"},
                         {"role": "user", "content": "How are you?"}]
        record_id: Unique identifier for tracking this record in output
        max_tokens: Maximum tokens to generate (default: 1024)
        system_prompt: Optional system prompt
        temperature: Sampling temperature (0.0-1.0)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        stop_sequences: List of stop sequences
    
    Returns:
        Dictionary formatted for Bedrock batch inference
    """
    # Convert simple string content to proper format
    formatted_messages = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        formatted_messages.append({
            "role": msg["role"],
            "content": content
        })
    
    model_input = {
        "max_tokens": max_tokens,
        "anthropic_version": "bedrock-2023-05-31",
        "messages": formatted_messages
    }
    
    # Add optional parameters
    if system_prompt:
        model_input["system"] = system_prompt
    if temperature is not None:
        model_input["temperature"] = temperature
    if top_p is not None:
        model_input["top_p"] = top_p
    if top_k is not None:
        model_input["top_k"] = top_k
    if stop_sequences:
        model_input["stop_sequences"] = stop_sequences
    
    record = {"modelInput": model_input}
    
    if record_id:
        record["recordId"] = record_id
    
    return record


# =============================================================================
# PART 2: FILE I/O UTILITIES
# =============================================================================

def save_batch_input_jsonl(records: List[dict], output_path: str) -> None:
    """
    Save batch input records to a JSONL file.
    
    Args:
        records: List of batch input dictionaries
        output_path: Path to output JSONL file
    """
    with open(output_path, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
    print(f"Saved {len(records)} records to {output_path}")


def upload_jsonl_in_chunks(
    local_file_path: str,
    bucket_name: str,
    s3_key_prefix: str,
    chunk_size: int = 50_000,
    region: str = 'us-east-1'
) -> List[str]:
    """
    Upload a large JSONL file to S3 in chunks.
    
    Args:
        local_file_path: Path to the local JSONL file
        bucket_name: Target S3 bucket name
        s3_key_prefix: S3 key prefix for chunked files
        chunk_size: Number of records per chunk (default: 50,000)
        region: AWS region
    
    Returns:
        List of uploaded S3 keys
    """
    s3_client = boto3.client('s3', region_name=region)
    
    chunk_number = 1
    total = 0
    uploaded_keys = []
    
    in_memory_chunk = io.BytesIO()

    with open(local_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_bytes = line.encode('utf-8')

            if total >= chunk_size:
                in_memory_chunk.seek(0)
                s3_object_key = os.path.join(s3_key_prefix, f"part_{chunk_number:04d}.jsonl")
                print(f"Uploading chunk {chunk_number} to s3://{bucket_name}/{s3_object_key}...")
                
                s3_client.upload_fileobj(in_memory_chunk, bucket_name, s3_object_key)
                uploaded_keys.append(s3_object_key)
                
                chunk_number += 1
                in_memory_chunk.close()
                in_memory_chunk = io.BytesIO()
                total = 0

            in_memory_chunk.write(line_bytes)
            total += 1

    # Upload final chunk
    if total > 0:
        in_memory_chunk.seek(0)
        s3_object_key = os.path.join(s3_key_prefix, f"part_{chunk_number:04d}.jsonl")
        print(f"Uploading final chunk {chunk_number} to s3://{bucket_name}/{s3_object_key}...")
        s3_client.upload_fileobj(in_memory_chunk, bucket_name, s3_object_key)
        uploaded_keys.append(s3_object_key)
    
    if not in_memory_chunk.closed:
        in_memory_chunk.close()
    
    print(f"\nUpload complete. {len(uploaded_keys)} chunks uploaded.")
    return uploaded_keys


def get_s3_object_keys(
    bucket_name: str,
    prefix: str,
    region: str = 'us-east-1'
) -> List[str]:
    """
    List S3 object keys matching a prefix.

    Args:
        bucket_name: S3 bucket name
        prefix: Key prefix to filter by
        region: AWS region

    Returns:
        List of object key filenames (without the prefix path)
    """
    s3_client = boto3.client('s3', region_name=region)
    paginator = s3_client.get_paginator('list_objects_v2')
    
    object_keys = []
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    
    print(f"Scanning s3://{bucket_name}/{prefix} ...")
    
    for page in page_iterator:
        if "Contents" in page:
            for obj in page["Contents"]:
                if obj['Key'] != prefix:
                    object_keys.append(obj['Key'].split('/')[-1])
    
    print(f"Found {len(object_keys)} files.")
    return object_keys


# =============================================================================
# PART 3: BATCH JOB MANAGER
# =============================================================================

class BedrockBatchJobManager:
    """
    Manages concurrent batch inference jobs on AWS Bedrock.
    
    Handles:
    - Job submission with concurrency limits
    - Status monitoring
    - Automatic retry on transient failures
    - Progress tracking
    """
    
    # Available Claude models on Bedrock (update as needed)
    CLAUDE_MODELS = {
        "haiku": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "sonnet": "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "opus": "us.anthropic.claude-opus-4-20250514-v1:0",
        # Add region-specific or versioned model IDs as needed
    }
    
    def __init__(
        self,
        max_concurrent_jobs: int = 10,
        input_bucket: str = None,
        output_bucket: str = None,
        role_arn: str = None,
        model_id: str = None,
        region: str = 'us-east-1'
    ):
        """
        Initialize the batch job manager.
        
        Args:
            max_concurrent_jobs: Maximum concurrent batch jobs (Bedrock limit is typically 10)
            input_bucket: S3 URI for input files (e.g., "s3://bucket/inputs/")
            output_bucket: S3 URI for output files (e.g., "s3://bucket/outputs/")
            role_arn: IAM role ARN for Bedrock to access S3
            model_id: Bedrock model ID (e.g., "us.anthropic.claude-haiku-4-5-20251001-v1:0")
            region: AWS region
        """
        self.max_concurrent_jobs = max_concurrent_jobs
        self.input_bucket = input_bucket
        self.output_bucket = output_bucket
        self.role_arn = role_arn
        self.model_id = model_id
        self.region = region
        
        self.bedrock = boto3.client('bedrock', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)
        
        # Job tracking
        self.active_jobs = {}
        self.completed_jobs = {}
        self.failed_jobs = {}
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_active_job_count(self) -> int:
        """Get count of currently active batch jobs."""
        response = self.bedrock.list_model_invocation_jobs()
        active_count = sum(
            1 for job in response['invocationJobSummaries']
            if job['status'] in ['InProgress', 'Submitted']
        )
        return active_count

    def wait_for_available_slot(self, check_interval: int = 30) -> None:
        """Block until a job slot becomes available."""
        while True:
            active_count = self.get_active_job_count()
            self.logger.info(f"Active jobs: {active_count}/{self.max_concurrent_jobs}")
            
            if active_count < self.max_concurrent_jobs:
                break
            
            self.logger.info(f"Max jobs reached. Waiting {check_interval}s...")
            time.sleep(check_interval)
    
    def create_batch_job(
        self,
        job_name: str,
        input_s3_key: str,
        timeout_hours: int = 24
    ) -> Optional[str]:
        """
        Create and submit a single batch inference job.
        
        Args:
            job_name: Unique name for the job
            input_s3_key: S3 key (filename) of input JSONL
            timeout_hours: Job timeout in hours
        
        Returns:
            Job ARN if successful, None otherwise
        """
        response = self.bedrock.create_model_invocation_job(
            jobName=job_name,
            roleArn=self.role_arn,
            modelId=self.model_id,
            inputDataConfig={
                's3InputDataConfig': {
                    's3Uri': os.path.join(self.input_bucket, input_s3_key),
                    's3InputFormat': 'JSONL'
                }
            },
            outputDataConfig={
                's3OutputDataConfig': {
                    's3Uri': os.path.join(self.output_bucket, job_name + "/")
                }
            },
            timeoutDurationInHours=timeout_hours,
            tags=[
                {'key': 'BatchGroup', 'value': 'batch-processing'},
                {'key': 'CreatedBy', 'value': 'BedrockBatchJobManager'},
                {'key': 'Timestamp', 'value': datetime.now().isoformat()}
            ]
        )
        
        job_arn = response['jobArn']
        self.active_jobs[job_arn] = {
            'name': job_name,
            'input_key': input_s3_key,
            'created_at': datetime.now(),
            'status': 'Submitted'
        }
        
        self.logger.info(f"Created job: {job_name} (ARN: {job_arn})")
        return job_arn
            
    def monitor_job_status(self, job_arn: str) -> str:
        """Get current status of a job."""
        try:
            response = self.bedrock.get_model_invocation_job(jobIdentifier=job_arn)
            return response['status']
        except Exception as e:
            self.logger.error(f"Error getting status for {job_arn}: {e}")
            return 'Unknown'
    
    def update_job_statuses(self) -> None:
        """Update status of all tracked jobs."""
        completed_arns = []
        
        for job_arn in list(self.active_jobs.keys()):
            status = self.monitor_job_status(job_arn)
            self.active_jobs[job_arn]['status'] = status
            
            if status in ['Completed', 'Failed', 'Stopped']:
                job_info = self.active_jobs[job_arn]
                job_info['completed_at'] = datetime.now()
                job_info['final_status'] = status
                
                if status == 'Completed':
                    self.completed_jobs[job_arn] = job_info
                    self.logger.info(f"Job completed: {job_info['name']}")
                else:
                    self.failed_jobs[job_arn] = job_info
                    self.logger.warning(f"Job {status.lower()}: {job_info['name']}")
                
                completed_arns.append(job_arn)
        
        for arn in completed_arns:
            del self.active_jobs[arn]
    
    def process_file_chunks(
        self,
        chunk_files: List[str],
        job_name_prefix: str = "batch-chunk",
        monitor_interval: int = 60,
        which: Optional[List[int]] = None
    ) -> Dict:
        """
        Process multiple input file chunks with concurrent job management.
        
        Args:
            chunk_files: List of input file names (keys) to process
            job_name_prefix: Prefix for job names
            monitor_interval: Seconds between status checks
            which: Optional list of indices to process (for partial runs)
        
        Returns:
            Summary dictionary with completion statistics
        """
        self.logger.info(f"Starting processing of {len(chunk_files)} chunks")
        self.logger.info(f"Max concurrent jobs: {self.max_concurrent_jobs}")
        
        submitted_jobs = []
        
        for i, chunk_file in enumerate(chunk_files):
            if which is not None and i not in which:
                continue
            
            while True:
                self.wait_for_available_slot()
                
                timestamp = int(time.time())
                job_name = f"{job_name_prefix}-{i+1:03d}-{timestamp}"
                
                try:
                    job_arn = self.create_batch_job(job_name, chunk_file)
                    if job_arn:
                        submitted_jobs.append(job_arn)
                        self.logger.info(f"Submitted job {i+1}/{len(chunk_files)}: {job_name}")
                        break
                except Exception as e:
                    self.logger.warning(f"Submission error: {e}. Retrying in 30s...")
                    time.sleep(30)
            
            time.sleep(2)  # Avoid API throttling
        
        # Monitor until completion
        self.logger.info("All jobs submitted. Monitoring progress...")
        
        while self.active_jobs:
            self.update_job_statuses()
            
            total = len(submitted_jobs)
            completed = len(self.completed_jobs)
            failed = len(self.failed_jobs)
            active = len(self.active_jobs)
            
            self.logger.info(
                f"Progress: {completed + failed}/{total} "
                f"(Completed: {completed}, Failed: {failed}, Active: {active})"
            )
            
            if self.active_jobs:
                time.sleep(monitor_interval)
        
        return {
            'total_submitted': len(submitted_jobs),
            'completed': len(self.completed_jobs),
            'failed': len(self.failed_jobs),
            'completed_jobs': self.completed_jobs,
            'failed_jobs': self.failed_jobs
        }

    def get_processing_summary(self) -> Dict:
        """Get detailed processing summary with timing statistics."""
        total_jobs = len(self.completed_jobs) + len(self.failed_jobs)
        
        if total_jobs == 0:
            return {"message": "No jobs processed yet"}
        
        processing_times = []
        for job_info in self.completed_jobs.values():
            if 'completed_at' in job_info and 'created_at' in job_info:
                duration = (job_info['completed_at'] - job_info['created_at']).total_seconds()
                processing_times.append(duration)
        
        avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        return {
            'total_jobs': total_jobs,
            'completed': len(self.completed_jobs),
            'failed': len(self.failed_jobs),
            'success_rate': len(self.completed_jobs) / total_jobs * 100,
            'average_processing_time_seconds': avg_time,
            'average_processing_time_minutes': avg_time / 60,
            'failed_job_details': [
                {'name': job['name'], 'status': job['final_status']}
                for job in self.failed_jobs.values()
            ]
        }


# =============================================================================
# PART 4: OUTPUT PARSING UTILITIES
# =============================================================================

def parse_bedrock_output(output_s3_uri: str) -> List[Dict]:
    """
    Parse Bedrock batch output files.
    
    Args:
        output_s3_uri: S3 URI pattern for output files 
                       (e.g., "s3://bucket/outputs/*/*.jsonl.out")
    
    Returns:
        List of parsed output records with record_id and response text
    
    Note: Requires polars for efficient parsing. Install with: pip install polars
    """
    try:
        import polars as pl
    except ImportError:
        raise ImportError("polars is required for output parsing. Install with: pip install polars")
    
    df = pl.scan_ndjson(output_s3_uri)
    df = df.select(
        pl.col("recordId").alias("record_id"),
        pl.col("modelOutput").struct.field("content").list.eval(
            pl.element().struct.field("text")
        ).list.join(" ").alias("response_text")
    ).collect()
    
    return df.to_dicts()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # ---------------------------------------------------------
    # EXAMPLE: Simple text summarization batch job
    # ---------------------------------------------------------
    
    # Step 1: Define your prompt formatting function
    def format_summarization_prompt(record: dict) -> str:
        """Example: Format a summarization prompt."""
        return f"""Please summarize the following text in 2-3 sentences:

{record['text']}

Summary:"""
    
    # Step 2: Prepare your input data
    sample_data = [
        {"id": "doc_001", "text": "This is the first document to summarize..."},
        {"id": "doc_002", "text": "This is the second document to summarize..."},
        {"id": "doc_003", "text": "This is the third document to summarize..."},
    ]
    
    # Step 3: Create batch input records
    batch_inputs = [
        create_llm_batch_input(
            prompt=format_summarization_prompt(record),
            record_id=record["id"],
            max_tokens=256,
            system_prompt="You are a concise summarization assistant."
        )
        for record in sample_data
    ]
    
    # Step 4: Save to local JSONL file
    output_file = "batch_input.jsonl"
    save_batch_input_jsonl(batch_inputs, output_file)
    
    print("\n--- Example batch input record ---")
    print(json.dumps(batch_inputs[0], indent=2))
    
    # ---------------------------------------------------------
    # EXAMPLE: Submit to Bedrock (uncomment to run)
    # ---------------------------------------------------------
    """
    # Configuration
    BUCKET = "your-bucket-name"
    INPUT_PREFIX = "batch-inputs/"
    OUTPUT_PREFIX = "batch-outputs/"
    ROLE_ARN = "arn:aws:iam::123456789:role/your-bedrock-role"
    
    # Upload chunks to S3
    upload_jsonl_in_chunks(
        local_file_path=output_file,
        bucket_name=BUCKET,
        s3_key_prefix=INPUT_PREFIX,
        chunk_size=50000
    )
    
    # Get list of uploaded chunk files
    chunk_files = get_s3_object_keys(BUCKET, INPUT_PREFIX + "part_")
    
    # Initialize manager
    manager = BedrockBatchJobManager(
        max_concurrent_jobs=8,
        input_bucket=f"s3://{BUCKET}/{INPUT_PREFIX}",
        output_bucket=f"s3://{BUCKET}/{OUTPUT_PREFIX}",
        role_arn=ROLE_ARN,
        model_id=BedrockBatchJobManager.CLAUDE_MODELS["haiku"]
    )
    
    # Process all chunks
    results = manager.process_file_chunks(
        chunk_files=chunk_files,
        job_name_prefix="summarization",
        monitor_interval=60
    )
    
    # Print summary
    summary = manager.get_processing_summary()
    print(f"Success rate: {summary['success_rate']:.1f}%")
    print(f"Average time: {summary['average_processing_time_minutes']:.1f} min")
    """