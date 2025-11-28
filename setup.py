#!/usr/bin/env python3
"""
Bias Analysis Pro - Setup Script
"""

import subprocess
import sys
import os

def install_requirements():
    """Install all required packages"""
    print("🚀 Installing Bias Analysis Pro requirements...")
    
    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("✅ All requirements installed successfully!")
        print("\n📋 Next steps:")
        print("1. Create a .env file with your API keys")
        print("2. Run: streamlit run app.py")
        print("3. Open http://localhost:8501 in your browser")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_requirements()