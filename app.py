import modal
import time
import random
import json
import os
import sys
from pathlib import Path
import subprocess
import shutil
import traceback

app = modal.App("ragen-github-webshop")

# 基础镜像配置
base_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "build-essential", "cmake")
    .pip_install(
    "torch==1.12.1",
    "transformers==4.25.1",
    "accelerate==0.19.0",
    "numpy==1.21.6",
    "requests==2.28.2",
    "PyYAML==6.0",
    "urllib3==1.26.16",
    "tqdm==4.65.0",
    "flask==1.1.4",
    "flask-cors==3.0.10",
    "scikit-learn==1.1.3",
    "pandas==1.5.3",
    "beautifulsoup4==4.9.3",
    "matplotlib==3.5.3",
    "seaborn==0.11.2",
    "gym==0.21.0",
    "selenium==4.1.0"
    )
    .run_commands(
        "git config --global http.postBuffer 1048576000"
    )
)

volume = modal.Volume.from_name("ragen-models", create_if_missing=True)

@app.function(
    image=base_image,
    gpu="A10G",
    timeout=86400,
    volumes={"/root/models": volume},
    secrets=[modal.Secret.from_name("my-huggingface-secret")]
)
def train_from_github():
    """使用真实WebShop环境的训练流程"""
    repo_url = "https://github.com/luyang963/learngit.git"
    work_dir = Path("/root/learngit")

    try:
        if work_dir.exists():
            shutil.rmtree(work_dir)

        subprocess.run(
            ["git", "clone", repo_url, str(work_dir)],
            capture_output=True, text=True, check=True
        )
        print("✅ GitHub repository cloned successfully")
    except Exception as e:
        print(f"❌ Git clone failed: {e}")
        return {"status": "error", "message": f"Git clone failed: {e}"}

    os.chdir(work_dir)

    # 修正 WebShop 路径大小写问题
    webshop_path = work_dir / "webshop"  # 小写 webshop
    if str(webshop_path) not in sys.path:
        sys.path.insert(0, str(webshop_path))
        print(f"🔧 Added WebShop path: {webshop_path}")

    # 确保项目根目录在 sys.path
    if str(work_dir) not in sys.path:
        sys.path.insert(0, str(work_dir))
        print(f"🔧 Added project root: {work_dir}")

    # 测试能否成功导入 WebShop 环境
    try:
        from webshop.web_agent_site.envs import web_agent_site_env
        print("✅ Imported web_agent_site_env successfully")
        print("Available classes:", [x for x in dir(web_agent_site_env) if 'Env' in x or 'env' in x.lower()])
    except Exception as e:
        print(f"❌ Failed to import web_agent_site_env: {e}")

    try:
        from webshop.web_agent_site.envs import web_agent_text_env
        print("✅ Imported web_agent_text_env successfully")
        print("Available classes:", [x for x in dir(web_agent_text_env) if 'Env' in x or 'env' in x.lower()])
    except Exception as e:
        print(f"❌ Failed to import web_agent_text_env: {e}")

    # 使用真实 WebShop 环境训练
    try:
        print("🎯 Using REAL WebShop environment...")

        from ragen.train_ragen_apo import RAGENWebShopTrainer

        config_path = "configs/webshop_config.yaml"
        trainer = RAGENWebShopTrainer(config_path)
        trainer.train()

        return {
            "status": "completed",
            "message": "Real WebShop training completed successfully",
            "environment": "real_webshop"
        }

    except Exception as e:
        print(f"❌ Real WebShop training failed: {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.function(
    image=base_image,
    volumes={"/root/models": volume}
)
def download_results():
    """下载训练结果"""
    from pathlib import Path
    import shutil

    print("📥 Downloading training results...")

    volume_path = Path("/root/models")
    local_path = Path(".")

    if not volume_path.exists():
        return {"status": "error", "message": "No data in shared volume"}

    downloaded_files = []
    for item in volume_path.iterdir():
        if item.is_file():
            shutil.copy2(item, local_path / item.name)
            downloaded_files.append(item.name)
            print(f"  ✅ Downloaded: {item.name}")

    return {"status": "success", "files": downloaded_files}


if __name__ == "__main__":
    with app.run():
        train_from_github.remote()
