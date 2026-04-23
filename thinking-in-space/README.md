### Installation

```bash
cd thinking-in-space

conda create --name vsibench python=3.10
conda activate vsibench
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -e .
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install transformers==4.40.0 peft==0.10.0 google-generativeai google-genai huggingface_hub[hf_xet]
```

### 

### Evaluation

We provide an evaluation scripts. You can simply run the following code to start your evaluation.

```bash
bash eval_vlm_3r.sh
```

