import subprocess
import sys

def instal_ffmpeg():
    print("Starting Ffmpeg installation...")
    
    subprocess.check_call([sys.executable, "-m", "pip"
                            "install", "--upgrade", "pip"])
    
    subprocess.check_call([sys.executable, "-m", "pip"
                            "install", "--upgrade", "setuptools"])
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip"
                                "install", "ffmpeg-python"])
        print("Ffmpeg-python installed")
    except subprocess.CalledProcessError as e:
        print("Ffmpeg installation failed {e}")
        
    try:
        subprocess.check_call([
            "wget",
            "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz",
            "-0", "/tmp/ffmpeg.tar.xz"
        ])
        
        subprocess.check_call(
            ["tar", "-xf", "/tmp/ffmpeg.tar.xz", "-C", "/tmp/"])
        
        ffmpeg_path = subprocess.check_output(
            ["tar", "-xf", "/tmp", "-name", "ffmpeg", "-type", "f"],
            text=True
        ).strip()
        
        subprocess.check_call(["cp", ffmpeg_path, "/usr/local/bin/ffmpeg"])
        
        subprocess.check_call(["chmod", "+x", "/usr/local/bin/ffmpeg"])
        
        print("Installed static Ffmpeg binary")
        
    except subprocess.CalledProcessError as e:
        print("Failed to install static FFmpeg {e}")
    
    try:
        result = subprocess.run(["ffmpeg", "--version"], capture_output=True, text=True, check=True)
        print("FFmpeg version:")
        print(result.stderr)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FFmpeg installation verification failed")
        return False