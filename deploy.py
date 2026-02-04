"""
Deploy Vespa Data Catalog Application

Deploys the semantic router application for data catalog metadata.
Architecture: BGE M3 embeddings + hybrid ranking + tree structure
"""

import os
import tempfile
from vespa.application import Vespa
from vespa_app import app_package
import requests

# Custom port mapping (host port 8090 -> container port 8080)
VESPA_PORT = 8090
CONFIG_PORT = 19071


def deploy_to_existing_container():
    """Deploy application to existing Vespa container."""
    print("Deploying Data Catalog application to Vespa...")
    print(f"  Schema: metadata")
    print(f"  Embeddings: 1024-dim (BGE M3 compatible)")
    print(f"  Ranking profiles: semantic, lexical, hybrid, alias_heavy")
    
    # Create a temporary directory with the application package
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export the application package
        app_package.to_files(tmpdir)
        
        # Create zip file for deployment
        import shutil
        zip_path = os.path.join(tmpdir, "application")
        shutil.make_archive(zip_path, 'zip', tmpdir)
        
        # Deploy via config server API
        deploy_url = f"http://localhost:{CONFIG_PORT}/application/v2/tenant/default/prepareandactivate"
        
        with open(f"{zip_path}.zip", 'rb') as f:
            response = requests.post(
                deploy_url,
                headers={"Content-Type": "application/zip"},
                data=f.read()
            )
        
        if response.status_code == 200:
            print("\n✅ Application deployed successfully!")
        else:
            print(f"\nDeployment response: {response.status_code}")
            print(response.text)
            
    # Return Vespa application handle
    app = Vespa(url=f"http://localhost:{VESPA_PORT}")
    print(f"Application URL: {app.url}")
    return app


def main():
    return deploy_to_existing_container()


if __name__ == "__main__":
    main()
