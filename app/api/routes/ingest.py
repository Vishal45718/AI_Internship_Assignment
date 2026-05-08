"""
app/api/routes/ingest.py — Document ingestion endpoints.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from app.core.dependencies import get_ingestion_pipeline, get_vector_store
from app.models.requests import IngestRequest, DeleteRequest
from app.models.responses import IngestResponse, IndexStatsResponse, ErrorResponse
from configs.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["Ingestion"])


@router.post(
    "",
    response_model=IngestResponse,
    summary="Ingest documents from file paths",
    description="Ingest one or more documents by providing their absolute file paths.",
)
async def ingest_files(
    request: IngestRequest,
    pipeline=Depends(get_ingestion_pipeline),
) -> IngestResponse:
    """Ingest documents by providing server-side file paths."""
    try:
        result = pipeline.ingest_files(request.file_paths)
        return IngestResponse(
            status="success" if not result.failed_files else "partial",
            files_processed=result.success_count,
            chunks_created=result.total_chunks,
            failed_files=result.failed_files,
            message=(
                f"Successfully ingested {result.success_count} file(s) "
                f"({result.total_chunks} chunks created)."
                + (f" {result.failure_count} file(s) failed." if result.failure_count else "")
            ),
        )
    except Exception as exc:
        logger.exception("Ingestion error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/upload",
    response_model=IngestResponse,
    summary="Upload and ingest files",
    description="Upload files directly via multipart form and ingest them immediately.",
)
async def upload_and_ingest(
    files: list[UploadFile] = File(...),
    pipeline=Depends(get_ingestion_pipeline),
) -> IngestResponse:
    """Accept file uploads, save to data dir, then ingest."""
    settings = get_settings()
    saved_paths: list[Path] = []
    failed: list[str] = []

    for upload in files:
        filename = upload.filename or "unknown_file"
        target = settings.data_dir / filename
        try:
            content = await upload.read()
            target.write_bytes(content)
            saved_paths.append(target)
            logger.info("Saved uploaded file: %s (%d bytes)", filename, len(content))
        except Exception as exc:
            logger.error("Failed to save '%s': %s", filename, exc)
            failed.append(filename)

    if saved_paths:
        result = pipeline.ingest_files([str(p) for p in saved_paths])
        return IngestResponse(
            status="success" if not result.failed_files else "partial",
            files_processed=result.success_count,
            chunks_created=result.total_chunks,
            failed_files=failed + result.failed_files,
            message=(
                f"Ingested {result.success_count} of {len(files)} uploaded file(s). "
                f"{result.total_chunks} chunks created."
            ),
        )
    else:
        return IngestResponse(
            status="error",
            files_processed=0,
            chunks_created=0,
            failed_files=failed,
            message="All uploaded files failed to save.",
        )


@router.get(
    "/stats",
    response_model=IndexStatsResponse,
    summary="Get index statistics",
)
async def index_stats(store=Depends(get_vector_store)) -> IndexStatsResponse:
    """Return statistics about the current vector index."""
    sources = store.list_sources()
    settings = get_settings()
    return IndexStatsResponse(
        total_chunks=store.count(),
        total_documents=len(sources),
        documents=sources,
        collection_name=settings.chroma_collection_name,
        persist_dir=str(settings.chroma_persist_dir),
    )


@router.delete(
    "/document",
    summary="Remove a document from the index",
    description="Remove all chunks belonging to a specific source document.",
)
async def delete_document(
    request: DeleteRequest,
    store=Depends(get_vector_store),
) -> JSONResponse:
    """Delete all chunks for a given source file from the vector index."""
    deleted = store.delete_by_source(request.source_file)
    if deleted == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No chunks found for source file: '{request.source_file}'",
        )
    return JSONResponse(
        content={
            "status": "success",
            "source_file": request.source_file,
            "chunks_deleted": deleted,
        }
    )


@router.post(
    "/reset",
    summary="Reset the entire vector index",
    description="⚠️ Destructive: deletes ALL indexed chunks. Use with caution.",
)
async def reset_index(store=Depends(get_vector_store)) -> JSONResponse:
    """Reset the vector store collection (delete all chunks)."""
    store.reset()
    return JSONResponse(
        content={"status": "success", "message": "Vector index has been reset."}
    )
