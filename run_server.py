#!/usr/bin/env python3
import uvicorn
import argparse
from src.utils.logging import setup_logging

def main():
    parser = argparse.ArgumentParser(description="Run RAG Backend Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    setup_logging()
    
    print(f"Starting API Server on http://{args.host}:{args.port}")
    uvicorn.run("backend.app.main:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()
