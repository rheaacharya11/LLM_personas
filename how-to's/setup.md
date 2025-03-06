# Setting Up PyTorch with CUDA on Harvard FAS RC

This guide provides step-by-step instructions to set up a virtual environment with CUDA-enabled PyTorch on Harvard FAS RC.

---

## **1. Load Required Modules**
Before creating an environment, load the necessary modules:
```bash
module load python/3.10.9-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01
```

If you are on a **login node**, you must request a **GPU session** before proceeding.

```bash
srun --partition=gpu_requeue --gres=gpu:1 --mem=16G --time=1:00:00 --pty /bin/bash
```
Verify that the GPU is available:
```bash
nvidia-smi
```

---

## **2. Create and Activate a Virtual Environment**
Create a new virtual environment in your home directory:
```bash
python -m venv ~/my_llama_python
source ~/my_llama_python/bin/activate
```

---

## **3. Install PyTorch with CUDA**
Install CUDA-enabled PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify installation:
```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
```
✅ **Expected Output:** PyTorch version (`2.x.x+cu121`), `True` for CUDA availability.

---

## **4. Install Hugging Face Transformers**
Install additional dependencies:
```bash
pip install transformers accelerate
```

---

## **5. Running Your Code**
Before running any script, activate the environment:
```bash
source ~/my_llama_python/bin/activate
```

Create a SLURM job script (`submit_job.sh`) to run jobs on a GPU:
```bash
#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-00:10
#SBATCH -p gpu_requeue
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -o myoutput_%j.out
#SBATCH -e myerrors_%j.err

module load python/3.10.9-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01

source ~/my_llama_python/bin/activate
python your_script.py
```

Submit the job:
```bash
sbatch submit_job.sh
```

---

## **6. Troubleshooting**
### **PyTorch Not Detecting CUDA (`False` Output)**
```bash
module list  # Ensure CUDA is loaded
srun --partition=gpu_requeue --gres=gpu:1 --mem=16G --time=1:00:00 --pty /bin/bash  # Start a GPU session
```

### **Virtual Environment Issues**
If the environment is broken, delete and recreate it:
```bash
rm -rf ~/my_llama_python
python -m venv ~/my_llama_python
source ~/my_llama_python/bin/activate
```

---

This setup ensures a CUDA-enabled PyTorch environment for efficient deep learning workloads on FAS RC. 🚀
