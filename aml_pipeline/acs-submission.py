import sys
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml import load_component, load_environment
from azureml.rag.utils.deployment import infer_deployment
from azureml.rag.utils.connections import get_connection_by_id_v2
from azure.ai.ml import dsl, Input
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import Input, Output
from azure.ai.ml.entities._job.pipeline._io import PipelineInput
from typing import Optional
import json

sys.path.append('../chunk_caption_component/')
sys.path.append('../enhanced_doc_analyzer_component/')


aoai_connection_name = "admro-m6gjoqlt-switzerlandnorth."
acs_connection_name = "beyondtext"
data_set_name = "raw_data_ds"
asset_name = "qknows_aoai_acs_mlindex"
doc_intelligence_connection_name = "beyondtext-doc-intelligence"
vision_deploy_name = "gpt-4"
aoai_embedding_model_name = "text-embedding-ada-002"

acs_config = {
    "index_name": asset_name,
}

experiment_name = "acs-embedding"

# Get workspace
ml_client = MLClient.from_config(
    credential=DefaultAzureCredential()
)


ml_registry = MLClient(credential=DefaultAzureCredential(), registry_name="azureml")

# Reads input folder of files containing chunks and their metadata as batches, in parallel, and generates embeddings for each chunk. Output format is produced and loaded by `azureml.rag.embeddings.EmbeddingContainer`.
generate_embeddings_component = ml_registry.components.get(
    "llm_rag_generate_embeddings", label="latest"
)
# Reads an input folder produced by `azureml.rag.embeddings.EmbeddingsContainer.save()` and pushes all documents (chunk, metadata, embedding_vector) into an Azure Cognitive Search index. Writes an MLIndex yaml detailing the index and embeddings model information.
update_acs_index_component = ml_registry.components.get(
    "llm_rag_update_acs_index", label="latest"
)
# Takes a uri to a storage location where an MLIndex yaml is stored and registers it as an MLIndex Data asset in the AzureML Workspace.
register_mlindex_component = ml_registry.components.get(
    "llm_rag_register_mlindex_asset", label="latest"
)

# Load components and environment
analyzer_component = load_component(source="./enhanced_doc_analyzer_component/doc_analyzer_component.yaml")
chunk_caption_index = load_component(source="./chuck_caption_component/chuck_caption_component.yaml")

aoai_connection_id = ml_client.connections.get(aoai_connection_name).id
aoai_connection = get_connection_by_id_v2(aoai_connection_id)



embeddings_model_uri = f"azure_open_ai://deployment/{aoai_embedding_model_name}/model/{aoai_embedding_model_name}"
# embeddings_model = "hugging_face://model/sentence-transformers/all-mpnet-base-v2"
embeddings_model = embeddings_model_uri


doc_intelligence_connection = ml_client.connections.get(doc_intelligence_connection_name)
acs_connection = ml_client.connections.get(acs_connection_name)

# Get the data asset with version
raw_papers = ml_client.data.get(data_set_name, version="1")
# Create Input object for the data
pdf_input = Input(type=AssetTypes.URI_FOLDER, path=raw_papers.path)


def optional_pipeline_input_provided(input: Optional[PipelineInput]):
    """Checks if optional pipeline inputs are provided."""
    return input is not None and input._data is not None

def use_automatic_compute(component, instance_count=1, instance_type="Standard_E8s_v3"):
    """Configure input `component` to use automatic compute with `instance_count` and `instance_type`.

    This avoids the need to provision a compute cluster to run the component.
    """
    component.set_resources(
        instance_count=instance_count,
        instance_type=instance_type,
        properties={"compute_specification": {"automatic": True}},
    )
    return component

@dsl.pipeline(
    description="Combined document analysis and azure AI search indexing pipeline",
    default_compute="serverless"
)
def document_processing_pipeline(
    # Document Analyzer inputs
    
    pdf_folder,
    asset_name: str,
    acs_config: str, 
    acs_connection_id: str,
    doc_intel_connection_id: str,
    confidence_threshold: float = 0.5,
    min_length: int = 10,
    overlap_threshold: float = 0.5,
    ignore_roles: str = "pageFooter,footnote,pageHeader",
    vision_deployment_name: str = "gpt-4",
    embeddings_model: str = "hugging_face://model/sentence-transformers/all-mpnet-base-v2",
    embeddings_container=None,
    aoai_connection_id: str = None,
    # Compute settings
    analyzer_compute: str = "gpu-cluster",
    indexer_compute: str = "cpu-cluster"

):
    # Document Analyzer step
    analysis_job = analyzer_component(
        input_folder=pdf_folder,
        doc_intel_connection_id=doc_intel_connection_id,
        confidence_threshold=confidence_threshold,
        min_length=min_length,
        overlap_threshold=overlap_threshold,
        ignore_roles=ignore_roles
    )
    analysis_job.compute = analyzer_compute

    # Chunk Caption Index step
    # Using the output from document analyzer as input
    chunk_caption_job = chunk_caption_index(
        input_folder=analysis_job.outputs.output_dir,
        azure_openai_connection_id=aoai_connection_id,
        vision_deployment_name=vision_deployment_name,
    )
    chunk_caption_job.compute = indexer_compute

    generate_embeddings = generate_embeddings_component(
        chunks_source=chunk_caption_job.outputs.output_folder,
        embeddings_container=embeddings_container,
        embeddings_model=embeddings_model,
    )
    use_automatic_compute(generate_embeddings)
    if optional_pipeline_input_provided(aoai_connection_id):
        generate_embeddings.environment_variables[
            "AZUREML_WORKSPACE_CONNECTION_ID_AOAI"
        ] = aoai_connection_id
    if optional_pipeline_input_provided(embeddings_container):
        # If provided, `embeddings_container` is expected to be a URI to folder, the folder can be empty.
        # Each sub-folder is generated by a `create_embeddings_component` run and can be reused for subsequent embeddings runs.
        generate_embeddings.outputs.embeddings = Output(
            type="uri_folder", path=f"{embeddings_container.path}/{{name}}"
        )

    # `update_acs_index` takes the Embedded data produced by `generate_embeddings` and pushes it into an Azure Cognitive Search index.
    update_acs_index = update_acs_index_component(
        embeddings=generate_embeddings.outputs.embeddings, acs_config=acs_config
    )
    use_automatic_compute(update_acs_index)
    if optional_pipeline_input_provided(acs_connection_id):
        update_acs_index.environment_variables[
            "AZUREML_WORKSPACE_CONNECTION_ID_ACS"
        ] = acs_connection_id

    register_mlindex = register_mlindex_component(
        storage_uri=update_acs_index.outputs.index, asset_name=asset_name
    )
    use_automatic_compute(register_mlindex)
    return {
        "mlindex_asset_uri": update_acs_index.outputs.index,
        "mlindex_asset_id": register_mlindex.outputs.asset_id,
        "analyzer_output": analysis_job.outputs.output_dir,
        "final_output": chunk_caption_job.outputs.output_folder
    }

# Create pipeline
pipeline = document_processing_pipeline(
    # Document Analyzer params
    pdf_folder=pdf_input,
    asset_name=asset_name,
    doc_intel_connection_id=doc_intelligence_connection.id,
    acs_config=json.dumps(acs_config),
    acs_connection_id=acs_connection.id,
    confidence_threshold=0.3,
    min_length=15,
    overlap_threshold=0.7,
    ignore_roles="pageFooter,footnote,pageHeader",
    
    # Chunk Caption Index params
    aoai_connection_id=aoai_connection_id,
    vision_deployment_name=vision_deploy_name,
    embeddings_model=embeddings_model,
    embeddings_container=None,
    
    # Compute settings
    analyzer_compute="gpu-cluster",
    indexer_compute="cpu-cluster"
)

# These are added so that in progress index generations can be listed in UI, this tagging is done automatically by UI.
pipeline.properties["azureml.mlIndexAssetName"] = asset_name
pipeline.properties["azureml.mlIndexAssetKind"] = "acs"
pipeline.properties["azureml.mlIndexAssetSource"] = "raw_papers"

# Submit the pipeline
run = ml_client.jobs.create_or_update(
    pipeline,
    experiment_name=experiment_name,
    tags={"type": "sample-document-processing"}
)



