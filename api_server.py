# FastAPI web interface for the AI Query System 

import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel
    fastapi_available = True
except ImportError:
    print("FastAPI not available, using basic HTTP server")
    fastapi_available = False

import sys
sys.path.append('src')

from ai_query_system import AIQuerySystem


# Pydantic models for FastAPI
if fastapi_available:
    class QueryRequest(BaseModel):
        question: str
        top_k: Optional[int] = 3

    class FeedbackRequest(BaseModel):
        query: str
        response: str
        helpful: bool
        comments: Optional[str] = ""


# Initialize the AI Query System
query_system = AIQuerySystem()


def create_fastapi_app():
    """Create FastAPI application"""
    app = FastAPI(
        title="AI Query System",
        description="A RAG-based chatbot system that answers questions using document knowledge",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize the query system on startup"""
        try:
            query_system.initialize()
        except Exception as e:
            print(f"Failed to initialize query system: {e}")
    
    @app.get("/")
    async def root():
        """Root endpoint with basic info"""
        return {
            "message": "Welcome to AI Query System API",
            "version": "1.0.0",
            "endpoints": {
                "query": "/query",
                "system_info": "/system/info",
                "documents": "/system/documents",
                "feedback": "/feedback"
            }
        }
    
    @app.post("/query")
    async def query_endpoint(request: QueryRequest) -> Dict[str, Any]:
        """Process a query and return response"""
        try:
            result = query_system.query(
                request.question, 
                top_k=request.top_k
            )
            return {
                "success": True,
                "data": result
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/system/info")
    async def system_info():
        """Get system information"""
        try:
            info = query_system.get_system_info()
            return {
                "success": True,
                "data": info
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/system/documents")
    async def list_documents():
        """List available documents"""
        try:
            docs = query_system.list_documents()
            return {
                "success": True,
                "data": {
                    "documents": docs,
                    "count": len(docs)
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/feedback")
    async def submit_feedback(request: FeedbackRequest):
        """Submit feedback for a query response"""
        try:
            query_system.add_feedback(
                request.query,
                request.response,
                request.helpful,
                request.comments
            )
            return {
                "success": True,
                "message": "Feedback submitted successfully"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


def create_simple_server():
    """Create a simple HTTP server as fallback"""
    import http.server
    import socketserver
    import urllib.parse
    import json
    
    class QueryHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {
                    "message": "AI Query System Simple Server",
                    "note": "Install FastAPI for full functionality"
                }
                self.wfile.write(json.dumps(response).encode())
            elif self.path == '/static/index.html' or self.path == '/index.html':
                try:
                    with open('static/index.html', 'r', encoding='utf-8') as f:
                        content = f.read()
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(content.encode())
                except FileNotFoundError:
                    self.send_error(404, "File not found")
            else:
                super().do_GET()
        
        def do_POST(self):
            if self.path == '/query':
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                try:
                    data = json.loads(post_data.decode('utf-8'))
                    question = data.get('question', '')
                    
                    if not question:
                        self.send_error(400, "Question is required")
                        return
                    
                    result = query_system.query(question)
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    
                    response = {"success": True, "data": result}
                    self.wfile.write(json.dumps(response).encode())
                    
                except Exception as e:
                    self.send_error(500, str(e))
            else:
                self.send_error(404, "Not found")
    
    return QueryHandler


def main():
    """Main function to start the server"""
    import sys
    
    # Initialize the query system
    try:
        query_system.initialize()
        print("Query system initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize query system: {e}")
        return
    
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("Invalid port number")
            return
    
    if fastapi_available:
        import uvicorn
        app = create_fastapi_app()
        print(f"Starting FastAPI server on http://localhost:{port}")
        print("API documentation available at http://localhost:{port}/docs")
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        import socketserver
        print(f"Starting simple HTTP server on http://localhost:{port}")
        print("Note: Install FastAPI and uvicorn for full functionality")
        
        handler = create_simple_server()
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"Server running at http://localhost:{port}")
            httpd.serve_forever()


if __name__ == "__main__":
    main()