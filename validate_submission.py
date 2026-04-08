import subprocess
import sys
import requests
import os

# ===== CHANGE THIS TO YOUR SPACE URL =====
PING_URL = "https://huggingface.co/spaces/Tan200514/guardiannet-final"
REPO_DIR = "."

def check_space():
    print("Checking Hugging Face Space...")
    try:
        r = requests.get(f"{PING_URL}/reset", timeout=10)
        if r.status_code == 200:
            print("✅ Space reachable (200 OK)")
            return True
        else:
            print(f"❌ Space returned {r.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot reach Space: {e}")
        return False

def check_docker():
    print("Checking Docker build...")
    try:
        subprocess.run(["docker", "build", "-t", "test", "."], check=True, cwd=REPO_DIR, timeout=300)
        print("✅ Docker image built successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Docker build failed")
        return False
    except FileNotFoundError:
        print("❌ Docker not found. Install Docker Desktop.")
        return False

def check_openenv():
    print("Checking openenv validate...")
    try:
        subprocess.run(["openenv", "validate", REPO_DIR], check=True, timeout=30)
        print("✅ openenv validate passed")
        return True
    except subprocess.CalledProcessError:
        print("❌ openenv validate failed")
        return False
    except FileNotFoundError:
        print("❌ openenv-core not installed. Run: pip install openenv-core")
        return False

if __name__ == "__main__":
    print("=== OpenEnv Submission Validator ===\n")
    ok = True
    if not check_space():
        ok = False
    if not check_docker():
        ok = False
    if not check_openenv():
        ok = False
    if ok:
        print("\n🎉 All checks passed! Your submission is ready.")
    else:
        print("\n❌ Some checks failed. Fix the issues above and try again.")
    sys.exit(0 if ok else 1)