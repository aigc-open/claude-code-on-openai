from . import app
import uvicorn

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Run with: uvicorn server:app --reload --host 0.0.0.0 --port 8082")
        sys.exit(0)
    
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="info")