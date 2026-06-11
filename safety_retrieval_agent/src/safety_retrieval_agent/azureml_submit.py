"""Azure ML command-job submission helpers for the Safety Retrieval Agent.

This version keeps the large build artifacts only in the stable AML datastore
path. It does not run the old datastore-to-datastore sync step.
"""
from __future__ import annotations

from .config import Settings


def _import_azureml():
    try:
        from azure.ai.ml import MLClient, Input, Output, command
        from azure.ai.ml.entities import Environment
        from azure.identity import DefaultAzureCredential
    except ImportError as exc:  # pragma: no cover - import guidance only
        raise ImportError(
            "Azure ML submission dependencies are missing. Install them locally with:\n"
            "    pip install -r requirements_azureml_submit.txt\n"
            "The Azure ML runtime environment itself is defined in environments/safety_retrieval_agent_cpu.yml."
        ) from exc
    return MLClient, Input, Output, command, Environment, DefaultAzureCredential


def get_ml_client(settings: Settings):
    MLClient, _Input, _Output, _command, _Environment, DefaultAzureCredential = _import_azureml()
    credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)
    return MLClient(
        credential=credential,
        subscription_id=settings.aml_subscription_id,
        resource_group_name=settings.aml_resource_group,
        workspace_name=settings.aml_workspace_name,
    )


def ensure_environment(ml_client, settings: Settings) -> str:
    _MLClient, _Input, _Output, _command, Environment, _DefaultAzureCredential = _import_azureml()
    env_ref = f"{settings.aml_environment_name}:{settings.aml_environment_version}"
    if not bool(settings.aml_register_environment):
        print(f"[AML] Using configured environment without registration: {env_ref}", flush=True)
        return env_ref
    conda_file = settings.azureml_conda_env_path()
    if not conda_file.exists():
        raise FileNotFoundError(f"Azure ML conda environment file not found: {conda_file}")
    env = Environment(
        name=settings.aml_environment_name,
        version=settings.aml_environment_version,
        image=settings.aml_base_image,
        conda_file=str(conda_file),
        description="CPU environment for Safety Retrieval Agent embedding/index jobs.",
    )
    print(f"[AML] Creating/updating environment: {env_ref}", flush=True)
    ml_client.environments.create_or_update(env)
    return env_ref


def _thread_exports(settings: Settings) -> str:
    cores = int(settings.aml_cpu_cores_per_node)
    return " && ".join(
        [
            f"export OMP_NUM_THREADS={cores}",
            f"export MKL_NUM_THREADS={cores}",
            f"export OPENBLAS_NUM_THREADS={cores}",
            f"export NUMEXPR_NUM_THREADS={cores}",
            "export TOKENIZERS_PARALLELISM=false",
            "export SAFETY_RETRIEVAL_FORCE_CPU_EMBEDDING=true",
            "export SAFETY_RETRIEVAL_EMBEDDING_DEVICE=cpu",
            "export SAFETY_RETRIEVAL_CPU_THREAD_COUNT=0",
            f"export SAFETY_RETRIEVAL_TORCH_INTEROP_THREADS={max(1, min(2, cores))}",
        ]
    )


def _prepared_input(settings: Settings, Input):
    if settings.aml_prepared_output_uri:
        print(f"[AML] Using prepared output URI as input: {settings.aml_prepared_output_uri}", flush=True)
        return Input(type="uri_folder", path=settings.aml_prepared_output_uri, mode=settings.aml_input_mode)
    if bool(settings.aml_use_local_output_dir_as_input):
        if not settings.output_dir.exists():
            raise FileNotFoundError(
                f"Local output_dir does not exist and aml_use_local_output_dir_as_input=True: {settings.output_dir}\n"
                "Run scripts/00_prepare_knowledge_base.py first, or set SAFETY_RETRIEVAL_AML_PREPARED_OUTPUT_URI."
            )
        print(f"[AML] Uploading/mounting local prepared output_dir as input: {settings.output_dir}", flush=True)
        return Input(type="uri_folder", path=str(settings.output_dir), mode=settings.aml_input_mode)
    return None


def submit_embedding_chunks_job(settings: Settings):
    _MLClient, Input, Output, command, _Environment, _DefaultAzureCredential = _import_azureml()
    ml_client = get_ml_client(settings)
    env_ref = ensure_environment(ml_client, settings)

    inputs = {}
    prepared = _prepared_input(settings, Input)
    prepared_expr = "${{outputs.output_dir}}"
    if prepared is not None:
        inputs["prepared_output_dir"] = prepared
        prepared_expr = "${{inputs.prepared_output_dir}}"

    outputs = {"output_dir": Output(type="uri_folder", path=settings.aml_output_uri, mode=settings.aml_output_mode)}
    shell = _thread_exports(settings)
    shell += " && export SAFETY_RETRIEVAL_OUTPUT_DIR=\"${{outputs.output_dir}}\""
    shell += f" && export SAFETY_RETRIEVAL_PREPARED_OUTPUT_DIR=\"{prepared_expr}\""
    shell += " && python scripts/01a_generate_embedding_chunks.py"

    job = command(
        code=str(settings.project_root),
        command=shell,
        inputs=inputs,
        outputs=outputs,
        environment=env_ref,
        compute=settings.aml_compute_name,
        experiment_name=settings.aml_experiment_name,
        display_name=settings.aml_embedding_job_display_name,
    )
    print(f"[AML] Submitting embedding chunk job to compute={settings.aml_compute_name}", flush=True)
    created = ml_client.jobs.create_or_update(job)
    print(f"[AML] Submitted job: {created.name}", flush=True)
    print(f"[AML] Stable output URI: {settings.aml_output_uri}", flush=True)
    try:
        print(f"[AML] Studio URL: {created.studio_url}", flush=True)
    except Exception:
        pass
    return created


def submit_index_build_job(settings: Settings):
    _MLClient, Input, Output, command, _Environment, _DefaultAzureCredential = _import_azureml()
    ml_client = get_ml_client(settings)
    env_ref = ensure_environment(ml_client, settings)

    inputs = {}
    copy_existing = ""
    if bool(settings.aml_index_download_existing_output_first):
        inputs["existing_output_dir"] = Input(type="uri_folder", path=settings.aml_output_uri, mode="download")
        copy_existing = " && cp -R \"${{inputs.existing_output_dir}}\"/. \"${{outputs.output_dir}}\"/ || true"

    outputs = {"output_dir": Output(type="uri_folder", path=settings.aml_output_uri, mode=settings.aml_output_mode)}
    shell = _thread_exports(settings)
    shell += " && export SAFETY_RETRIEVAL_OUTPUT_DIR=\"${{outputs.output_dir}}\""
    shell += copy_existing
    shell += " && python scripts/01b_build_indexes_from_chunks.py"

    job = command(
        code=str(settings.project_root),
        command=shell,
        inputs=inputs,
        outputs=outputs,
        environment=env_ref,
        compute=settings.aml_compute_name,
        experiment_name=settings.aml_experiment_name,
        display_name=settings.aml_index_job_display_name,
    )
    print(f"[AML] Submitting index build job to compute={settings.aml_compute_name}", flush=True)
    created = ml_client.jobs.create_or_update(job)
    print(f"[AML] Submitted job: {created.name}", flush=True)
    print(f"[AML] Stable output URI: {settings.aml_output_uri}", flush=True)
    try:
        print(f"[AML] Studio URL: {created.studio_url}", flush=True)
    except Exception:
        pass
    return created


def submit_full_batch_job(settings: Settings):
    """Submit one AML command job for embedding + index creation only.

    Large artifacts remain in settings.aml_artifact_output_uri / aml_output_uri.
    The old sync-to-workspace step is intentionally not run in this version.
    Runtime scripts read directly from the datastore through azureml-fsspec.
    """
    _MLClient, Input, Output, command, _Environment, _DefaultAzureCredential = _import_azureml()
    ml_client = get_ml_client(settings)
    env_ref = ensure_environment(ml_client, settings)

    inputs = {}
    prepared = _prepared_input(settings, Input)
    prepared_expr = "${{outputs.output_dir}}"
    if prepared is not None:
        inputs["prepared_output_dir"] = prepared
        prepared_expr = "${{inputs.prepared_output_dir}}"

    copy_existing = ""
    if bool(getattr(settings, "aml_full_copy_existing_output_first", False)):
        inputs["existing_output_dir"] = Input(type="uri_folder", path=settings.aml_output_uri, mode="download")
        copy_existing = " && cp -R \"${{inputs.existing_output_dir}}\"/. \"${{outputs.output_dir}}\"/ || true"

    outputs = {"output_dir": Output(type="uri_folder", path=settings.aml_output_uri, mode=settings.aml_output_mode)}
    shell = _thread_exports(settings)
    shell += " && export SAFETY_RETRIEVAL_OUTPUT_DIR=\"${{outputs.output_dir}}\""
    shell += f" && export SAFETY_RETRIEVAL_PREPARED_OUTPUT_DIR=\"{prepared_expr}\""
    shell += copy_existing
    shell += " && python scripts/01a_generate_embedding_chunks.py"
    shell += " && python scripts/01b_build_indexes_from_chunks.py"

    job = command(
        code=str(settings.project_root),
        command=shell,
        inputs=inputs,
        outputs=outputs,
        environment=env_ref,
        compute=settings.aml_compute_name,
        experiment_name=settings.aml_experiment_name,
        display_name=settings.aml_full_job_display_name,
    )
    print(f"[AML] Submitting full embedding/index job to compute={settings.aml_compute_name}", flush=True)
    print(f"[AML] Artifact output URI: {settings.aml_output_uri}", flush=True)
    created = ml_client.jobs.create_or_update(job)
    print(f"[AML] Submitted full job: {created.name}", flush=True)
    try:
        print(f"[AML] Studio URL: {created.studio_url}", flush=True)
    except Exception:
        pass
    return created


def stream_job_until_complete(settings: Settings, job_name: str) -> None:
    """Stream an Azure ML job until completion from the local submit process."""
    ml_client = get_ml_client(settings)
    print(f"[AML] Streaming job until completion: {job_name}", flush=True)
    ml_client.jobs.stream(job_name)


def submit_download_runtime_artifacts_job(settings: Settings):
    """Submit a cloud-side runtime-artifact copy/check job.

    This job is useful for validating that the artifact folder can be read and
    copied inside an AML command-job environment. It does not write into the
    interactive VS Code filesystem. For local VS Code testing, run:

        python scripts/01c_download_runtime_artifacts.py
    """
    _MLClient, Input, Output, command, _Environment, _DefaultAzureCredential = _import_azureml()
    ml_client = get_ml_client(settings)
    env_ref = ensure_environment(ml_client, settings)

    inputs = {
        "artifact_input_dir": Input(type="uri_folder", path=settings.aml_output_uri, mode="download"),
    }
    outputs = {
        "downloaded_runtime_dir": Output(
            type="uri_folder",
            path=settings.aml_output_uri.rstrip("/") + "/runtime-download-check/",
            mode=settings.aml_output_mode,
        )
    }
    shell = _thread_exports(settings)
    shell += " && export SAFETY_RETRIEVAL_ARTIFACT_AZUREML_URI=\"${{inputs.artifact_input_dir}}\""
    shell += " && export SAFETY_RETRIEVAL_OUTPUT_DIR=\"${{outputs.downloaded_runtime_dir}}\""
    shell += " && export SAFETY_RETRIEVAL_ARTIFACT_READ_MODE=local"
    shell += " && python scripts/01c_download_runtime_artifacts.py"

    job = command(
        code=str(settings.project_root),
        command=shell,
        inputs=inputs,
        outputs=outputs,
        environment=env_ref,
        compute=settings.aml_compute_name,
        experiment_name=settings.aml_experiment_name,
        display_name=settings.aml_download_job_display_name,
    )
    print(f"[AML] Submitting runtime artifact download/check job to compute={settings.aml_compute_name}", flush=True)
    created = ml_client.jobs.create_or_update(job)
    print(f"[AML] Submitted job: {created.name}", flush=True)
    try:
        print(f"[AML] Studio URL: {created.studio_url}", flush=True)
    except Exception:
        pass
    return created
